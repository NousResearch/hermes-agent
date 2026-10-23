// Path rules for the `ctx.files` door (plugin-files.ts): how a plugin-supplied
// relative path is split, how containment is compared, and which folders can
// never be granted. Pure helpers plus filesystem identity checks; no state.

import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { realpathOrNull, refuse } from './plugin-files-ops'

const REL_PATH_MAX = 1024
const SEGMENT_MAX = 255
// Plugin ids become a folder name under the shared data root, so only a plain
// slug may reach the join (no separators, no leading dot, no `..`, and no
// trailing dot, which Windows drops: `foo.` would share `foo`'s folder).
const PLUGIN_ID_RE = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?$/
// Windows device names open the device, not a file, with any extension.
const WIN_RESERVED_RE = /^(?:con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$|clock\$)(?:\.|$)/i
// Kernel pseudo-filesystems: their files are "regular" with size 0 and can
// hold the app's own environment (`/proc/self/environ`).
const SYSTEM_ROOTS = ['/proc', '/sys', '/dev']

// The same filesystems by type, wherever they are mounted (a container's
// `/host/proc`, a chroot's `proc`): proc, sysfs, debugfs, tracefs,
// securityfs, cgroup, cgroup2, bpf, efivarfs, configfs, pstore, selinuxfs.
const PSEUDO_FS_TYPES = new Set([
  0x9fa0, 0x62656572, 0x64626720, 0x74726163, 0x73636673, 0x27e0eb, 0x63677270, 0xcafe4a11, 0xde5e81e4, 0x62656570,
  0x6165676c, 0xf97cff8c
])

const WSL_SHARE_RE = /^\\\\(?:wsl\.localhost|wsl\$)\\[^\\]+/i
// Top-level WSL folders that are not files of the distro: its kernel views
// (process environments included) and its mounts of the Windows drives.
const WSL_SYSTEM_DIRS = new Set(['dev', 'mnt', 'proc', 'run', 'sys'])

const within = (child: string, parent: string): boolean => {
  const rel = path.relative(parent, child)

  return rel === '' || (rel !== '..' && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel))
}

export function pathTools(platform: NodeJS.Platform) {
  // Every comparison errs toward refusing. Containment ("may the plugin use
  // this?") compares realpaths exactly, so a case-sensitive volume's `vault`
  // is never mistaken for the granted `Vault`. Protection ("is this Hermes
  // data?") also matches case-insensitively where volumes usually are, so a
  // differently-cased spelling of a protected path is still refused.
  const caseless = platform === 'win32' || platform === 'darwin'
  const inside = (child: string, parent: string) => within(child, parent)

  const insideLoose = (child: string, parent: string) =>
    within(child, parent) || (caseless && within(child.toLowerCase(), parent.toLowerCase()))

  const same = (a: string, b: string) => path.resolve(a) === path.resolve(b)

  // Split a plugin-supplied relative path into plain segments, refusing (not
  // normalizing) anything that could name a place outside the folder.
  const segments = (raw: unknown): string[] => {
    if (typeof raw !== 'string' || raw.length > REL_PATH_MAX || raw.includes('\0')) {
      return refuse('invalid-path', 'Path must be a relative path inside the folder.')
    }

    if (path.posix.isAbsolute(raw) || path.win32.isAbsolute(raw) || /^[a-zA-Z]:/.test(raw)) {
      return refuse('invalid-path', 'Path must be relative to the folder, not absolute.')
    }

    const parts = raw.split(/[\\/]+/).filter(part => part && part !== '.')

    for (const part of parts) {
      // `..namedfork` opens a macOS file's forks under another name
      // (`.env/..namedfork/data` is `.env`).
      if (part === '..' || part.length > SEGMENT_MAX || part.toLowerCase() === '..namedfork') {
        return refuse('invalid-path', 'Path may not leave the folder.')
      }

      // Windows drops trailing dots/spaces (`.. ` is `..`), reads `:` as a
      // stream, and opens a device for `CON`, `NUL.txt`…
      if (platform === 'win32' && (part.includes(':') || /[. ]$/.test(part) || WIN_RESERVED_RE.test(part))) {
        return refuse('invalid-path', 'Path segment is not valid on Windows.')
      }
    }

    return parts
  }

  return { inside, insideLoose, same, segments }
}

export const pluginKey = (pluginId: unknown): string =>
  typeof pluginId === 'string' && PLUGIN_ID_RE.test(pluginId) && !WIN_RESERVED_RE.test(pluginId)
    ? pluginId
    : refuse('invalid-plugin', 'Plugin id is not a plain slug.')

// Case-insensitive filesystems would give `Plexus` and `plexus` one folder; an
// id with capitals gets a suffix (`~` never appears in an id) to keep them apart.
export function dataDirName(pluginId: string): string {
  return pluginId === pluginId.toLowerCase()
    ? pluginId
    : `${pluginId.toLowerCase()}~${createHash('sha256').update(pluginId).digest('hex').slice(0, 8)}`
}

export const systemRoots = (platform: NodeJS.Platform): string[] => (platform === 'win32' ? [] : SYSTEM_ROOTS)

export async function realRoots(roots: string[]): Promise<string[]> {
  const resolved = await Promise.all(
    roots.filter(Boolean).map(async root => (await realpathOrNull(root)) ?? path.resolve(root))
  )

  return [...new Set(resolved)]
}

/**
 * A WSL share path that is Hermes data or not the distro's own files: the
 * distro root, `/home`, a home dir and `/root` each hold a `~/.hermes`, and
 * anything under a `.hermes` segment is one (Windows Desktop cannot know which
 * WSL user runs the gateway, so every WSL home counts); `/proc`, `/sys`,
 * `/dev` and `/run` are the distro's kernel views, and `/mnt` reaches the
 * Windows drives (and this app's own data) by a second spelling.
 */
export function wslProtectedPath(candidate: string): boolean {
  const share = WSL_SHARE_RE.exec(candidate)

  if (!share) {
    return false
  }

  const rest = candidate
    .slice(share[0].length)
    .split('\\')
    .filter(Boolean)
    .map(part => part.toLowerCase())

  const holdsHome =
    rest.length === 0 ||
    (rest.length === 1 && (rest[0] === 'home' || rest[0] === 'root')) ||
    (rest.length === 2 && rest[0] === 'home')

  return holdsHome || WSL_SYSTEM_DIRS.has(rest[0]) || rest.includes('.hermes')
}

const LOOPBACK_HOSTS = new Set(['localhost', '::1', '[::1]', '0--1.ipv6-literal.net'])

/**
 * A Windows path that reaches this machine's own drives a second way, which a
 * realpath comparison cannot match to the protected folders: a device path
 * (`\\?\C:\`, `\\.\`, `\\?\GLOBALROOT\…` shadow copies) or a share served by
 * this machine (`\\localhost\C$`, `\\127.0.0.1\…`, its own name or address).
 */
export function loopbackShare(candidate: string): boolean {
  const unc = /^\\\\\?\\UNC\\/i.test(candidate) ? `\\\\${candidate.slice(8)}` : candidate

  if (/^\\\\[?.]\\/.test(unc)) {
    return true
  }

  const host = /^\\\\([^\\]+)/.exec(unc)?.[1].split('@')[0].toLowerCase()

  if (!host) {
    return false
  }

  const own = new Set([os.hostname().toLowerCase()])

  for (const addresses of Object.values(os.networkInterfaces())) {
    for (const address of addresses ?? []) {
      own.add(address.address.toLowerCase())
    }
  }

  return LOOPBACK_HOSTS.has(host) || /^127\./.test(host) || own.has(host) || own.has(host.split('.')[0])
}

/** True when `target` lives on a kernel pseudo-filesystem (Linux). */
export async function onPseudoFilesystem(target: string): Promise<boolean> {
  const stats = await fs.promises.statfs(target).catch(() => null)

  return stats !== null && PSEUDO_FS_TYPES.has(Number(stats.type))
}

const identity = async (target: string): Promise<null | string> => {
  const stat = await fs.promises.stat(target).catch(() => null)

  return stat ? `${stat.dev}:${stat.ino}` : null
}

// The identity of a path and of every folder above it.
async function identityChain(target: string): Promise<string[]> {
  const ids: string[] = []

  for (let current = target; ; current = path.dirname(current)) {
    const id = await identity(current)

    if (id) {
      ids.push(id)
    }

    if (path.dirname(current) === current) {
      return ids
    }
  }
}

/**
 * Overlap by filesystem identity, not spelling: a bind mount, a macOS firmlink
 * (`/System/Volumes/Data/Users/…`) or a case-insensitive mount is a second
 * path to the same folder that a realpath string comparison would miss.
 */
export async function overlapsByIdentity(candidate: string, roots: string[]): Promise<boolean> {
  const self = await identity(candidate)
  const chain = await identityChain(candidate)

  for (const root of roots) {
    const rootId = await identity(root)

    if (rootId && (chain.includes(rootId) || (self && (await identityChain(root)).includes(self)))) {
      return true
    }
  }

  return false
}

/**
 * True when `target`, or a folder between it and `root`, is one of `roots` by
 * filesystem identity: a bind mount of the Hermes home inside a granted folder
 * keeps an inside path but is the protected folder itself.
 */
export async function protectedIdentityWithin(target: string, root: string, roots: string[]): Promise<boolean> {
  const ids = new Set((await Promise.all(roots.map(identity))).filter((id): id is string => id !== null))

  for (let current = target; ; current = path.dirname(current)) {
    const id = await identity(current)

    if (id && ids.has(id)) {
      return true
    }

    if (current === root || path.dirname(current) === current) {
      return false
    }
  }
}

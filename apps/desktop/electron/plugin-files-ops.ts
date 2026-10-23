// The operations behind `ctx.files` (plugin-files.ts owns grants and path
// resolution). Each op gets a target already resolved inside its folder and
// re-applies the per-op rules: size caps, sensitive names, no op on the folder
// itself, regular files only, O_EXCL creation. Reads and writes go through a
// handle whose identity (dev/ino) must match the file that was checked, so a
// swap between the check and the open is refused rather than followed.

import fs from 'node:fs'
import path from 'node:path'

import { sensitiveFileBlockReason } from './hardening'
import type { PluginFilesErrorCode, PluginFilesFailure, PluginFilesRequest, PluginFilesResult } from './plugin-files'

const TEXT_MAX_BYTES = 4 * 1024 * 1024
const DATA_URL_MAX_BYTES = 16 * 1024 * 1024
const LIST_MAX_ENTRIES = 10_000
const READ_CHUNK_BYTES = 1024 * 1024
// Folders whose contents are keys; renaming one away from its name would put
// those keys out of reach of the name-based sensitive-file rules.
const SENSITIVE_DIRS = new Set(['.aws', '.gnupg', '.ssh'])
// Non-blocking so a FIFO or device that slipped in cannot park a main-process
// worker; no-follow so a link swapped in for an existing file is refused.
const READ_FLAGS = fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK ?? 0) | (fs.constants.O_NOFOLLOW ?? 0)
const UPDATE_FLAGS = fs.constants.O_RDWR | (fs.constants.O_NONBLOCK ?? 0) | (fs.constants.O_NOFOLLOW ?? 0)

const IMAGE_MIME: Record<string, string> = {
  '.avif': 'image/avif',
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp'
}

export const fail = (error: PluginFilesErrorCode, message: string): PluginFilesFailure => ({ ok: false, error, message })

export class Refusal extends Error {
  constructor(readonly failure: PluginFilesFailure) {
    super(failure.message)
  }
}

export const refuse = (error: PluginFilesErrorCode, message: string): never => {
  throw new Refusal(fail(error, message))
}

export const errnoCode = (error: unknown): string | undefined =>
  error && typeof error === 'object' && 'code' in error ? String((error as { code?: unknown }).code) : undefined

export async function lstatOrNull(target: string): Promise<fs.Stats | null> {
  try {
    return await fs.promises.lstat(target)
  } catch (error) {
    if (errnoCode(error) === 'ENOENT') {
      return null
    }

    throw error
  }
}

export async function realpathOrNull(target: string): Promise<null | string> {
  try {
    return await fs.promises.realpath(target)
  } catch (error) {
    const code = errnoCode(error)

    if (code === 'ENOENT' || code === 'ENOTDIR' || code === 'ELOOP') {
      return null
    }

    throw error
  }
}

// Default-ignorable code points (zero-width, bidi marks, BOM, soft hyphen,
// variation selectors), which HFS+ skips when it compares names.
const IGNORABLE_RANGES: [number, number][] = [
  [0xad, 0xad],
  [0x34f, 0x34f],
  [0x115f, 0x1160],
  [0x17b4, 0x17b5],
  [0x180b, 0x180f],
  [0x200b, 0x200f],
  [0x202a, 0x202e],
  [0x2060, 0x206f],
  [0x3164, 0x3164],
  [0xfe00, 0xfe0f],
  [0xfeff, 0xfeff],
  [0xffa0, 0xffa0]
]

const ignorable = (char: string) => {
  const code = char.codePointAt(0) ?? 0

  return IGNORABLE_RANGES.some(([low, high]) => code >= low && code <= high)
}

// The app's sensitive-file rules, checked on the name as given and as macOS
// and Windows may match it (Unicode case folding: `id_rſa` opens `id_rsa`
// there; HFS+ also ignores zero-width characters).
export function sensitiveReason(target: string): null | string {
  const folded = [...target.normalize('NFKC')]
    .filter(char => !ignorable(char))
    .join('')
    .toLowerCase()

  const reason = sensitiveFileBlockReason(target) ?? sensitiveFileBlockReason(folded)

  if (reason) {
    return reason
  }

  return SENSITIVE_DIRS.has(path.basename(folded)) ? 'That folder holds keys and cannot be used by a plugin.' : null
}

export const refuseSensitive = (...targets: string[]) => {
  for (const target of targets) {
    const reason = sensitiveReason(target)

    if (reason) {
      refuse('sensitive-file', reason)
    }
  }
}

const sameFile = (a: fs.Stats, b: fs.Stats) => a.dev === b.dev && a.ino === b.ino

// Only a plain, singly-linked file: a FIFO or device has no size to cap, and a
// hard link may be a second name for a file anywhere on the volume.
function refuseNonPlain(stat: fs.Stats) {
  if (!stat.isFile()) {
    refuse('not-a-file', 'That path is not a regular file.')
  }

  if (stat.nlink > 1) {
    refuse('protected', 'That file is hard-linked elsewhere, so the app cannot tell what it belongs to.')
  }
}

// The opened file must be the one that was checked AND still sit at the
// checked path inside the folder: a parent folder swapped for a link after
// `resolve` makes the same path open a file elsewhere.
async function refuseMoved(resolved: Resolved, opened: fs.Stats) {
  const real = await realpathOrNull(resolved.target)
  const now = real === resolved.target ? await lstatOrNull(real) : null

  if (!sameFile(opened, resolved.stat!) || !now || !sameFile(now, opened)) {
    refuse('outside-folder', 'The file changed while it was being opened.')
  }
}

// Rename, trash and list act on paths (Node has no dirfd-relative rename,
// unlink or readdir), so each re-checks right before (and list and rename
// right after) that the entry is still the checked one in the checked folder.
async function refuseChanged(resolved: Resolved) {
  const parent = resolved.parent === null ? null : await realpathOrNull(resolved.parent)
  const real = await realpathOrNull(resolved.entry)
  const now = real ? await lstatOrNull(real) : null

  if (parent !== resolved.parent || real !== resolved.target || !now || !sameFile(now, resolved.stat!)) {
    refuse('outside-folder', 'The folder changed while it was being used.')
  }
}

// Move `from` back to `to` only if nothing is at `to`: a file through a hard
// link plus unlink (the link refuses an existing name), a folder by rename
// after checking the name is free.
async function restoreWithoutOverwrite(from: string, to: string): Promise<boolean> {
  const moved = await lstatOrNull(from)

  if (!moved || (await lstatOrNull(to))) {
    return false
  }

  try {
    if (moved.isDirectory()) {
      await fs.promises.rename(from, to)
    } else {
      await fs.promises.link(from, to)
      await fs.promises.unlink(from)
    }

    return true
  } catch {
    return false
  }
}

// Read at most `maxBytes` through a handle that must still be the checked
// file. The cap counts bytes read, not the size stat reported (0 for a pseudo
// file), so a file is refused past the cap whatever it claims.
async function readChecked(resolved: Resolved, maxBytes: number): Promise<Buffer> {
  const handle = await fs.promises.open(resolved.target, READ_FLAGS)

  try {
    const opened = await handle.stat()
    await refuseMoved(resolved, opened)
    refuseNonPlain(opened)
    const chunks: Buffer[] = []
    let total = 0

    for (;;) {
      const chunk = Buffer.alloc(Math.min(READ_CHUNK_BYTES, maxBytes + 1 - total))
      const { bytesRead } = await handle.read(chunk, 0, chunk.length, total)

      if (bytesRead === 0) {
        return Buffer.concat(chunks, total)
      }

      chunks.push(chunk.subarray(0, bytesRead))
      total += bytesRead

      if (total > maxBytes) {
        return refuse('too-large', `File is larger than ${Math.round(maxBytes / (1024 * 1024))} MiB.`)
      }
    }
  } finally {
    await handle.close()
  }
}

// Write `text` to the checked target: an existing file through a no-follow
// handle that must be the checked file (and is only truncated after that),
// a new one created O_EXCL and then confirmed to sit where it was checked.
async function writeChecked(resolved: Resolved, text: string) {
  if (resolved.stat) {
    refuseNonPlain(resolved.stat)
    const handle = await fs.promises.open(resolved.target, UPDATE_FLAGS)

    try {
      const opened = await handle.stat()
      await refuseMoved(resolved, opened)
      refuseNonPlain(opened)
      await handle.truncate(0)
      await handle.writeFile(text, 'utf8')
    } finally {
      await handle.close()
    }

    return
  }

  const handle = await fs.promises.open(resolved.entry, 'wx')

  try {
    const created = await handle.stat()
    const real = await realpathOrNull(resolved.entry)
    const landed = real ? await lstatOrNull(real) : null

    if (real !== resolved.entry || !landed || !sameFile(landed, created)) {
      if (landed && sameFile(landed, created)) {
        await fs.promises.unlink(real!)
      }

      refuse('outside-folder', 'The folder changed while the file was being created.')
    }

    await handle.writeFile(text, 'utf8')
  } finally {
    await handle.close()
  }
}

function mimeFor(target: string): string {
  return IMAGE_MIME[path.extname(target).toLowerCase()] ?? 'application/octet-stream'
}

const toRel = (parts: string[]) => parts.join('/')

export interface Resolved {
  /** The folder's realpath. */
  root: string
  parts: string[]
  /** Realpath of the target's parent (null for the folder itself). */
  parent: null | string
  /** The entry as named (in the real parent), not following a final link. */
  entry: string
  /** What reads/writes act on: the entry's realpath (inside the folder). */
  target: string
  /** `lstat` of `target` (null when the entry does not exist yet). */
  stat: fs.Stats | null
}

export interface PluginFileOpsDeps {
  resolve: (pluginId: string, folder: unknown, rel: unknown) => Promise<Resolved>
  inside: (child: string, parent: string) => boolean
  segments: (raw: unknown) => string[]
  trashItem: (target: string) => Promise<void>
}

type PluginFileOp = (pluginId: string, req: any) => Promise<PluginFilesResult>

export function createPluginFileOps({
  resolve,
  inside,
  segments,
  trashItem
}: PluginFileOpsDeps): Record<PluginFilesRequest['op'], PluginFileOp> {
  const requireFile = (resolved: Resolved, maxBytes: number) => {
    if (!resolved.stat) {
      refuse('not-found', 'No such file.')
    }

    if (resolved.stat!.isDirectory()) {
      refuse('is-directory', 'That path is a folder.')
    }

    refuseNonPlain(resolved.stat!)

    if (resolved.stat!.size > maxBytes) {
      refuse('too-large', `File is larger than ${Math.round(maxBytes / (1024 * 1024))} MiB.`)
    }

    refuseSensitive(resolved.entry, resolved.target)
  }

  const requireChild = (resolved: Resolved) => {
    if (!resolved.parent) {
      refuse('invalid-path', 'This operation needs a path inside the folder, not the folder itself.')
    }
  }

  return {
    async list(pluginId, req) {
      const resolved = await resolve(pluginId, req.folder, req.path)

      if (!resolved.stat?.isDirectory()) {
        return fail(resolved.stat ? 'not-a-directory' : 'not-found', 'That path is not a folder.')
      }

      await refuseChanged(resolved)
      const dirents = await fs.promises.readdir(resolved.target, { withFileTypes: true })
      await refuseChanged(resolved)
      const entries: { name: string; path: string; isDirectory: boolean }[] = []

      for (const dirent of dirents.slice(0, LIST_MAX_ENTRIES)) {
        const full = path.join(resolved.target, dirent.name)
        let isDirectory = dirent.isDirectory()

        if (dirent.isSymbolicLink()) {
          const real = await realpathOrNull(full)

          if (!real || !inside(real, resolved.root)) {
            continue
          }

          isDirectory = (await fs.promises.stat(real)).isDirectory()
        }

        entries.push({ name: dirent.name, path: toRel([...resolved.parts, dirent.name]), isDirectory })
      }

      return { ok: true, entries, truncated: dirents.length > LIST_MAX_ENTRIES }
    },

    async stat(pluginId, req) {
      const { stat } = await resolve(pluginId, req.folder, req.path)

      return stat
        ? { ok: true, exists: true, isDirectory: stat.isDirectory(), size: stat.size, mtimeMs: stat.mtimeMs }
        : { ok: true, exists: false }
    },

    async readText(pluginId, req) {
      const resolved = await resolve(pluginId, req.folder, req.path)
      requireFile(resolved, TEXT_MAX_BYTES)
      const data = await readChecked(resolved, TEXT_MAX_BYTES)

      if (data.subarray(0, 8000).includes(0)) {
        return fail('binary', 'That file is binary; read it as a data URL.')
      }

      return { ok: true, text: data.toString('utf8'), size: data.length, mtimeMs: resolved.stat!.mtimeMs }
    },

    async readDataUrl(pluginId, req) {
      const resolved = await resolve(pluginId, req.folder, req.path)
      requireFile(resolved, DATA_URL_MAX_BYTES)
      const data = await readChecked(resolved, DATA_URL_MAX_BYTES)

      return { ok: true, url: `data:${mimeFor(resolved.target)};base64,${data.toString('base64')}` }
    },

    async writeText(pluginId, req) {
      const text = typeof req.text === 'string' ? req.text : ''

      if (Buffer.byteLength(text, 'utf8') > TEXT_MAX_BYTES) {
        return fail('too-large', 'Text is larger than 4 MiB.')
      }

      const resolved = await resolve(pluginId, req.folder, req.path)
      requireChild(resolved)

      if (resolved.stat?.isDirectory()) {
        return fail('is-directory', 'That path is a folder.')
      }

      if (resolved.stat && req.exclusive) {
        return fail('exists', 'A file with that name already exists.')
      }

      refuseSensitive(resolved.entry, resolved.target)
      await writeChecked(resolved, text)
      const stat = await fs.promises.stat(resolved.target)

      return { ok: true, path: toRel(resolved.parts), size: stat.size, mtimeMs: stat.mtimeMs }
    },

    async rename(pluginId, req) {
      const name = typeof req.name === 'string' ? req.name.trim() : ''

      if (segments(name).length !== 1 || /[\\/]/.test(name) || name === '.') {
        return fail('invalid-path', 'A new name is one plain file or folder name.')
      }

      const resolved = await resolve(pluginId, req.folder, req.path)
      requireChild(resolved)

      if (!resolved.stat) {
        return fail('not-found', 'No such file or folder.')
      }

      const destination = path.join(resolved.parent!, name)
      refuseSensitive(resolved.entry, resolved.target, destination)

      if (await lstatOrNull(destination)) {
        return fail('exists', `"${name}" already exists.`)
      }

      await refuseChanged(resolved)
      await fs.promises.rename(resolved.entry, destination)

      // If the folder was swapped in the last instant, put the moved entry
      // back, never over a file that has taken its old name meanwhile.
      const landedIn = await realpathOrNull(path.dirname(destination))
      const landed = await fs.promises.stat(destination).catch(() => null)

      if (landedIn !== resolved.parent || !landed || !sameFile(landed, resolved.stat!)) {
        const undone = await restoreWithoutOverwrite(destination, resolved.entry)
        refuse(
          'outside-folder',
          undone
            ? 'The folder changed during the rename, so it was undone.'
            : 'The folder changed during the rename and it could not be undone safely.'
        )
      }

      return { ok: true, path: toRel([...resolved.parts.slice(0, -1), name]) }
    },

    async trash(pluginId, req) {
      const resolved = await resolve(pluginId, req.folder, req.path)
      requireChild(resolved)

      if (!resolved.stat) {
        return fail('not-found', 'No such file or folder.')
      }

      refuseSensitive(resolved.entry, resolved.target)
      await refuseChanged(resolved)
      await trashItem(resolved.entry)

      return { ok: true }
    }
  }
}

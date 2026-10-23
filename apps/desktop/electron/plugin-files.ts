// The folder-scoped file door behind the plugin SDK's `ctx.files`. A plugin
// names a folder handle (`'data'` = its own data folder, or the id of a folder
// the user granted it through the app's picker) plus a RELATIVE path; this
// module, in the main process, decides what that may reach:
//
// - a relative path is split into plain segments; `..`, absolute paths, drive
//   letters, UNC paths and NUL bytes are refused, never normalized away;
// - the parent of every target is realpath'd and must sit inside the folder's
//   realpath, and an existing target that is a symlink is followed only when
//   its realpath stays inside (a dangling link is refused, so a write can
//   never create the link's outside target);
// - nothing may resolve into a protected root (the Hermes homes, which hold
//   config.yaml, .env, auth and every profile; the app's own data folder,
//   which holds every plugin's data folder and the grants record), and a
//   folder overlapping one in either direction (by path, by filesystem
//   identity, as a WSL home or a loopback share) can never be granted, and
//   no op reaches a kernel pseudo-filesystem (/proc, /sys, …) by path or by
//   filesystem type;
// - reads and writes are size-capped and refused, never truncated, and the
//   app's sensitive-file rules (.env, SSH/GPG keys, .npmrc…) hold inside a
//   granted folder too; a key folder (~/.ssh, ~/.gnupg) cannot be granted.
//
// Grants live in `<userData>/plugin-files/grants.json`, written only here.
// The renderer cannot mint one: `addGrant` is reached only from the native
// folder picker the user answers (plugin-files-ipc.ts).
//
// This bounds what the DOOR can reach. It is not a sandbox for plugin code:
// a renderer-realm plugin can still call the raw `window.hermesDesktop` file
// bridge, which is why catalog admission requires the door instead.

import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import {
  dataDirName,
  loopbackShare,
  onPseudoFilesystem,
  overlapsByIdentity,
  pathTools,
  pluginKey,
  protectedIdentityWithin,
  realRoots,
  systemRoots,
  wslProtectedPath
} from './plugin-files-guards'
import {
  createPluginFileOps,
  errnoCode,
  fail,
  lstatOrNull,
  realpathOrNull,
  Refusal,
  refuse,
  type Resolved,
  sensitiveReason
} from './plugin-files-ops'

export const PLUGIN_DATA_FOLDER = 'data'

const GRANT_ID_RE = /^[a-f0-9]{16}$/

export type PluginFilesErrorCode =
  | 'binary'
  | 'exists'
  | 'folder-changed'
  | 'folder-missing'
  | 'invalid-op'
  | 'invalid-path'
  | 'invalid-plugin'
  | 'io-error'
  | 'is-directory'
  | 'not-a-directory'
  | 'not-a-file'
  | 'not-found'
  | 'not-granted'
  | 'outside-folder'
  | 'protected'
  | 'sensitive-file'
  | 'too-large'

export interface PluginFilesFailure {
  ok: false
  error: PluginFilesErrorCode
  message: string
}

export type PluginFilesResult<T extends object = object> = ({ ok: true } & T) | PluginFilesFailure

export interface PluginFolderGrant {
  id: string
  name: string
  path: string
  grantedAt: string
}

export type PluginFilesRequest =
  | { op: 'list'; folder: string; path: string }
  | { op: 'readDataUrl'; folder: string; path: string }
  | { op: 'readText'; folder: string; path: string }
  | { op: 'rename'; folder: string; path: string; name: string }
  | { op: 'stat'; folder: string; path: string }
  | { op: 'trash'; folder: string; path: string }
  | { op: 'writeText'; folder: string; path: string; text: string; exclusive?: boolean }

export interface PluginFilesDeps {
  /** Electron's userData dir, read lazily (main sets it before `ready`). */
  userDataDir: () => string
  /** Hermes homes (the resolved HERMES_HOME and the platform defaults). */
  protectedRoots: () => string[]
  trashItem: (target: string) => Promise<void>
  platform?: NodeJS.Platform
}

export interface PluginFiles {
  grants: (pluginId: unknown) => Promise<PluginFolderGrant[]>
  addGrant: (pluginId: unknown, picked: string) => Promise<PluginFilesResult<{ folder: PluginFolderGrant }>>
  revoke: (pluginId: unknown, folderId?: unknown) => Promise<{ ok: true; removed: number }>
  run: (pluginId: unknown, request: PluginFilesRequest) => Promise<PluginFilesResult>
}

export function createPluginFiles(deps: PluginFilesDeps): PluginFiles {
  const platform = deps.platform ?? process.platform
  const { inside, insideLoose, same, segments } = pathTools(platform)
  const baseDir = () => path.join(deps.userDataDir(), 'plugin-files')
  const storeFile = () => path.join(baseDir(), 'grants.json')

  type Store = Record<string, PluginFolderGrant[]>
  let store: null | Store = null
  let loading: null | Promise<Store> = null
  let writes: Promise<unknown> = Promise.resolve()

  // Null prototype: a plugin id such as `constructor` is just a key.
  const fresh = (grants: unknown): Store =>
    Object.assign(Object.create(null), grants && typeof grants === 'object' ? grants : {})

  const readStore = async (): Promise<Store> => {
    // A missing or unreadable-as-JSON record is no grants (fail closed). Any
    // other read error is thrown, so a later write never replaces a record
    // it could not read.
    const raw = await fs.promises.readFile(storeFile(), 'utf8').catch(error => {
      if (errnoCode(error) === 'ENOENT') {
        return '{}'
      }

      throw error
    })

    try {
      return fresh((JSON.parse(raw) as { grants?: unknown } | null)?.grants)
    } catch {
      return fresh(null)
    }
  }

  // One initial read, shared; a read finishing late never replaces a record
  // a write already published.
  const load = async (): Promise<Store> => {
    if (store) {
      return store
    }

    loading ??= readStore().finally(() => {
      loading = null
    })
    const read = await loading
    store ??= read

    return store
  }

  // Serialized read-modify-write on a copy, saved with tmp + rename, and only
  // published once saved: a failed save changes nothing, now or after restart.
  const mutate = <T>(change: (grants: Store) => T): Promise<T> => {
    const next = writes.then(async () => {
      const grants = fresh(structuredClone({ ...(await load()) }))
      const result = change(grants)
      await fs.promises.mkdir(baseDir(), { recursive: true })
      const tmp = `${storeFile()}.${process.pid}.tmp`
      await fs.promises.writeFile(tmp, JSON.stringify({ version: 1, grants }, null, 2), { mode: 0o600 })
      await fs.promises.rename(tmp, storeFile())
      store = grants

      return result
    })

    writes = next.catch(() => undefined)

    return next
  }

  const protectedRoots = async (): Promise<string[]> =>
    realRoots([...deps.protectedRoots(), deps.userDataDir(), ...systemRoots(platform)])

  // A second spelling of this machine's files that no realpath comparison can
  // match: a WSL share or a loopback/device path on Windows.
  const secondPath = (real: string) => platform === 'win32' && (wslProtectedPath(real) || loopbackShare(real))

  // Kernel pseudo-filesystems by type, so one mounted anywhere is refused.
  const refusePseudo = async (real: string) => {
    if (platform === 'linux' && (await onPseudoFilesystem(real))) {
      refuse('protected', 'That is a kernel filesystem and cannot be used by a plugin.')
    }
  }

  const overlapsProtected = (real: string, roots: string[]) =>
    roots.some(root => insideLoose(real, root) || insideLoose(root, real)) || secondPath(real)

  // The folder a handle names, as a realpath, plus the protected roots an op
  // inside it must still avoid (none for the plugin's own data folder, whose
  // root already excludes every sibling).
  const openFolder = async (pluginId: string, folder: unknown): Promise<{ root: string; avoid: string[] }> => {
    const roots = await realRoots(deps.protectedRoots())

    if (folder === PLUGIN_DATA_FOLDER) {
      const dir = path.join(baseDir(), 'data', dataDirName(pluginId))
      await fs.promises.mkdir(dir, { recursive: true })

      return { root: await fs.promises.realpath(dir), avoid: roots }
    }

    const grant =
      typeof folder === 'string' && GRANT_ID_RE.test(folder)
        ? (await load())[pluginId]?.find(row => row.id === folder)
        : undefined

    if (!grant) {
      return refuse('not-granted', 'This plugin has no access to that folder.')
    }

    const real = await realpathOrNull(grant.path)

    if (!real) {
      return refuse('folder-missing', 'The granted folder no longer exists.')
    }

    // A grant is to the folder the user picked, not to whatever a path later
    // resolves to (a folder swapped for a link into another tree).
    if (!same(real, grant.path)) {
      return refuse('folder-changed', 'The granted folder was moved or replaced; grant it again.')
    }

    const avoid = await protectedRoots()

    if (overlapsProtected(real, avoid)) {
      return refuse('protected', 'That folder overlaps Hermes data and cannot be used by a plugin.')
    }

    await refusePseudo(real)

    return { root: real, avoid }
  }

  const resolve = async (pluginId: string, folder: unknown, rel: unknown): Promise<Resolved> => {
    const parts = segments(rel)
    const { root, avoid } = await openFolder(pluginId, folder)

    const guard = (real: string) => {
      if (!inside(real, root)) {
        refuse('outside-folder', 'That path leaves the folder.')
      }

      if (avoid.some(protectedRoot => insideLoose(real, protectedRoot)) || secondPath(real)) {
        refuse('protected', 'That path is Hermes data and cannot be used by a plugin.')
      }
    }

    // Hermes data mounted inside the folder keeps an inside path; its
    // filesystem identity gives it away.
    const refuseMounted = async (real: string) => {
      if (await protectedIdentityWithin(real, root, avoid)) {
        refuse('protected', 'That path is Hermes data mounted inside the folder and cannot be used by a plugin.')
      }
    }

    if (parts.length === 0) {
      return { root, parts, parent: null, entry: root, target: root, stat: await fs.promises.stat(root) }
    }

    const parent = await realpathOrNull(path.join(root, ...parts.slice(0, -1)))

    if (!parent) {
      return refuse('not-found', 'The parent folder does not exist.')
    }

    guard(parent)

    const entry = path.join(parent, parts[parts.length - 1])
    const lstat = await lstatOrNull(entry)

    if (!lstat) {
      await refuseMounted(parent)

      return { root, parts, parent, entry, target: entry, stat: null }
    }

    // Every existing entry is checked by its realpath: a symlink's target, and
    // the long, correctly-cased name behind a Windows short name (`ENV~1`).
    const target = await realpathOrNull(entry)

    if (!target) {
      return refuse('outside-folder', 'That link points nowhere inside the folder.')
    }

    guard(target)
    await refusePseudo(target)
    await refuseMounted(target)

    // The realpath itself, not followed: a link here now means the path was
    // swapped after it was resolved. The ops re-check it after they open.
    const stat = await fs.promises.lstat(target)

    if (stat.isSymbolicLink()) {
      return refuse('outside-folder', 'The file changed while it was being checked.')
    }

    return { root, parts, parent, entry, target, stat }
  }

  const OPS = createPluginFileOps({ resolve, inside, segments, trashItem: deps.trashItem })

  const answer = async <T>(work: () => Promise<T>): Promise<PluginFilesFailure | T> => {
    try {
      return await work()
    } catch (error) {
      if (error instanceof Refusal) {
        return error.failure
      }

      const code = errnoCode(error)

      return code === 'ENOENT' || code === 'ENOTDIR'
        ? fail('not-found', 'No such file or folder.')
        : code === 'EEXIST'
          ? fail('exists', 'A file with that name already exists.')
          : fail('io-error', error instanceof Error ? error.message : String(error))
    }
  }

  return {
    async grants(pluginId) {
      const key = typeof pluginId === 'string' ? pluginId : ''

      return [...((await load())[key] ?? [])]
    },

    addGrant: (pluginId, picked) =>
      answer(async () => {
        const key = pluginKey(pluginId)
        const real = await realpathOrNull(String(picked || ''))

        if (!real || !(await fs.promises.stat(real)).isDirectory()) {
          return fail('not-a-directory', 'Pick a folder.')
        }

        if ((await realRoots(systemRoots(platform))).some(root => insideLoose(real, root))) {
          return fail('protected', 'That is a system folder and cannot be shared with a plugin.')
        }

        const roots = await protectedRoots()

        if (overlapsProtected(real, roots) || (await overlapsByIdentity(real, roots))) {
          return fail('protected', 'That folder holds Hermes data and cannot be shared with a plugin.')
        }

        await refusePseudo(real)

        if (sensitiveReason(path.join(real, 'entry')) || sensitiveReason(real)) {
          return fail('protected', 'That folder holds keys or credentials and cannot be shared with a plugin.')
        }

        return mutate(grants => {
          const rows = (grants[key] ??= [])
          const existing = rows.find(row => same(row.path, real))

          if (existing) {
            return { ok: true as const, folder: { ...existing } }
          }

          const folder = {
            id: randomBytes(8).toString('hex'),
            name: path.basename(real) || real,
            path: real,
            grantedAt: new Date().toISOString()
          }

          rows.push(folder)

          return { ok: true as const, folder: { ...folder } }
        })
      }),

    revoke: (pluginId, folderId) =>
      mutate(grants => {
        const key = typeof pluginId === 'string' ? pluginId : ''
        const rows = grants[key] ?? []
        const kept = folderId === undefined ? [] : rows.filter(row => row.id !== folderId)

        if (kept.length) {
          grants[key] = kept
        } else {
          delete grants[key]
        }

        return { ok: true as const, removed: rows.length - kept.length }
      }),

    run: (pluginId, request) =>
      answer(async () => {
        const key = pluginKey(pluginId)
        const op = request && typeof request === 'object' ? request.op : undefined

        if (typeof op !== 'string' || !Object.prototype.hasOwnProperty.call(OPS, op)) {
          return fail('invalid-op', 'Unknown file operation.')
        }

        return OPS[op](key, request)
      })
  }
}

/**
 * `ctx.files` — the folder-scoped file door. A plugin reads and writes inside
 * its own data folder (`'data'`) and inside folders the user granted it
 * through the app's picker, by RELATIVE path. Electron's main process owns the
 * grants and checks every call (electron/plugin-files.ts); this side only
 * names the plugin, supplies the app's own picker copy, and keeps the
 * Capabilities ▸ Plugins grant list fresh. Every member resolves a result and
 * never throws, so an older shell reads as `{ ok: false, error: 'unavailable' }`.
 */

import { translateNow } from '@/i18n/runtime'
import { queryClient } from '@/lib/query-client'
import { notify } from '@/store/notifications'

import { $pluginRecords } from './plugins-store'

export const PLUGIN_FOLDER_GRANTS_KEY = ['plugin-folder-grants'] as const

export interface PluginFolder {
  /** Opaque handle to pass as `folder` to the other calls. */
  id: string
  /** The folder's name, for display. */
  name: string
  /** Absolute path the user picked, for display only — calls take `id`. */
  path: string
}

export interface PluginFileEntry {
  name: string
  /** Relative to the folder, `/`-separated; pass it back as `path`. */
  path: string
  isDirectory: boolean
}

export type PluginFilesResult<T extends object = object> =
  | ({ ok: true } & T)
  | { ok: false; error: string; message: string }

export interface PluginFiles {
  /** Handle of this plugin's private data folder (always available). */
  readonly dataFolder: 'data'
  /** Ask the user for a folder with the app's own picker. `folder` is null on
   *  cancel. A folder holding Hermes data is refused (`error: 'protected'`)
   *  and the app tells the user why. */
  requestFolder: () => Promise<PluginFilesResult<{ folder: PluginFolder | null }>>
  /** Folders the user has granted this plugin and not revoked. */
  folders: () => Promise<PluginFolder[]>
  /** Give a granted folder back (the user can also revoke it in Settings). */
  release: (folderId: string) => Promise<boolean>
  list: (folder: string, path?: string) => Promise<PluginFilesResult<{ entries: PluginFileEntry[]; truncated: boolean }>>
  stat: (
    folder: string,
    path: string
  ) => Promise<PluginFilesResult<{ exists: boolean; isDirectory?: boolean; mtimeMs?: number; size?: number }>>
  /** UTF-8 text up to 4 MiB; a binary or larger file is refused, never truncated. */
  readText: (folder: string, path: string) => Promise<PluginFilesResult<{ mtimeMs: number; size: number; text: string }>>
  /** A file (up to 16 MiB) as a `data:` URL, for image previews. */
  readDataUrl: (folder: string, path: string) => Promise<PluginFilesResult<{ url: string }>>
  /** Write up to 4 MiB of UTF-8 text. The parent folder must exist;
   *  `exclusive` refuses to replace an existing file. */
  writeText: (
    folder: string,
    path: string,
    text: string,
    options?: { exclusive?: boolean }
  ) => Promise<PluginFilesResult<{ mtimeMs: number; path: string; size: number }>>
  /** Rename in place (same parent folder); `name` is one plain name. */
  rename: (folder: string, path: string, name: string) => Promise<PluginFilesResult<{ path: string }>>
  /** Move to the OS trash (recoverable). The folder itself cannot be trashed. */
  trash: (folder: string, path: string) => Promise<PluginFilesResult>
}

const unavailable = { ok: false as const, error: 'unavailable', message: 'This Hermes Desktop build has no plugin file access.' }

const bridge = () => (typeof window === 'undefined' ? undefined : window.hermesDesktop?.pluginFiles)

const folderOf = (row: PluginFolder): PluginFolder => ({ id: row.id, name: row.name, path: row.path })

const refreshGrantList = () => void queryClient.invalidateQueries({ queryKey: PLUGIN_FOLDER_GRANTS_KEY })

/** Drop a desktop plugin's folder grants once it is really gone. Removing a
 *  package's agent half from one profile leaves the app-wide desktop half
 *  (and its grants) in place while another profile still installs it. A
 *  failed revoke rejects, so the caller can tell the user. */
export async function revokeGrantsIfUninstalled(pluginId: string): Promise<boolean> {
  if ($pluginRecords.get()[pluginId]) {
    return false
  }

  try {
    await bridge()?.revoke(pluginId)
  } finally {
    refreshGrantList()
  }

  return true
}

export function createPluginFiles(pluginId: string): PluginFiles {
  const run = async <T extends object>(request: Record<string, unknown>): Promise<PluginFilesResult<T>> => {
    const door = bridge()

    if (!door) {
      return unavailable
    }

    try {
      return (await door.op(pluginId, request)) as PluginFilesResult<T>
    } catch (error) {
      return { ok: false, error: 'io-error', message: error instanceof Error ? error.message : String(error) }
    }
  }

  return {
    dataFolder: 'data',

    async requestFolder() {
      const door = bridge()

      if (!door) {
        return unavailable
      }

      const name = $pluginRecords.get()[pluginId]?.name ?? pluginId

      try {
        const result = await door.request(pluginId, {
          buttonLabel: translateNow('skills.plugins.folderAccess.pickerButton'),
          title: translateNow('skills.plugins.folderAccess.pickerTitle', name)
        })

        if (!result.ok) {
          if (result.error === 'protected') {
            notify({ kind: 'warning', message: translateNow('skills.plugins.folderAccess.refusedProtected', name) })
          }

          return result
        }

        refreshGrantList()

        return { ok: true, folder: result.folder ? folderOf(result.folder) : null }
      } catch (error) {
        return { ok: false, error: 'io-error', message: error instanceof Error ? error.message : String(error) }
      }
    },

    folders: async () => {
      try {
        return ((await bridge()?.grants(pluginId)) ?? []).map(folderOf)
      } catch {
        return []
      }
    },

    async release(folderId) {
      try {
        const removed = ((await bridge()?.revoke(pluginId, folderId))?.removed ?? 0) > 0
        refreshGrantList()

        return removed
      } catch {
        return false
      }
    },

    list: (folder, path = '') => run({ op: 'list', folder, path }),
    stat: (folder, path) => run({ op: 'stat', folder, path }),
    readText: (folder, path) => run({ op: 'readText', folder, path }),
    readDataUrl: (folder, path) => run({ op: 'readDataUrl', folder, path }),
    writeText: (folder, path, text, options) =>
      run({ op: 'writeText', folder, path, text, exclusive: Boolean(options?.exclusive) }),
    rename: (folder, path, name) => run({ op: 'rename', folder, path, name }),
    trash: (folder, path) => run({ op: 'trash', folder, path })
  }
}

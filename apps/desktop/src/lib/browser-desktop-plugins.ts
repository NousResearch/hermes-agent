import type { HermesApiRequest, HermesPreviewFileChanged, HermesReadFileTextResult } from '@/global'

type PreviewWatchBridge = Pick<
  Window['hermesDesktop'],
  'onPreviewFileChanged' | 'stopPreviewFileWatch' | 'watchPreviewFile'
>

type DesktopPluginsBridge = Pick<
  Window['hermesDesktop'],
  'desktopPluginsRoot' | 'readDir' | 'readFileText' | 'readPluginSource'
> &
  PreviewWatchBridge

interface BrowserDesktopPluginsOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  /** The page's /api/fs readers, kept for every path outside the plugins folder. */
  files: Pick<Window['hermesDesktop'], 'readDir' | 'readFileText'>
  /** The page's preview watchers, kept for every path outside the plugins folder. */
  previews: PreviewWatchBridge
}

/** How often a visible tab asks for the plugin stamps while it watches a `plugin.js`. Native
 *  Desktop reloads on the fs event; a second here keeps the agent's edit-and-look loop tight. */
export const PLUGIN_WATCH_POLL_MS = 1_000

/** The runtime loader's disk door, answered by the server that serves this page: its app-level
 *  desktop-plugins folder (hermes_cli/web_routers/desktop_plugins.py). /api/fs is the wrong
 *  reader there: it follows the profile's SSH workspace and truncates at the preview cap. */
export function createBrowserDesktopPluginsBridge({
  api,
  files,
  previews
}: BrowserDesktopPluginsOptions): DesktopPluginsBridge {
  let root: null | string = null

  const get = <T>(route: string, path: string) =>
    api<T>({ path: `/api/desktop-plugins/${route}?${new URLSearchParams({ path })}` })

  // The server resolves the folder and builds every path it returns from it, so a path is inside
  // exactly when it starts with the root and a separator. The loader asks for the root first.
  const inside = (path: string) =>
    root !== null && (path === root || path.startsWith(`${root}/`) || path.startsWith(`${root}\\`))

  const readFile = (path: string) => get<HermesReadFileTextResult>('read', path)

  // A browser has no file watcher, so plugin.js watches poll one listing of every entry's stamp.
  // Concurrent asks (the loader binds a watch per plugin at once) share the request in flight.
  const watches = new Map<string, { path: string; stamp: string }>()
  const listeners = new Set<(payload: HermesPreviewFileChanged) => void>()
  let pending: null | Promise<Record<string, string>> = null
  let timer: null | number = null
  let nextId = 0

  const stamps = () =>
    (pending ??= api<{ stamps: Record<string, string> }>({ path: '/api/desktop-plugins/stamps' })
      .then(answer => answer.stamps)
      .finally(() => {
        pending = null
      }))

  const poll = async () => {
    if (document.visibilityState !== 'visible') {
      return
    }

    let current: Record<string, string>

    try {
      current = await stamps()
    } catch {
      return // A dropped request is not a change; the next tick asks again.
    }

    for (const [id, watch] of watches) {
      const stamp = current[watch.path] ?? ''

      if (stamp !== watch.stamp) {
        watch.stamp = stamp
        listeners.forEach(listener => listener({ id, path: watch.path, url: watch.path }))
      }
    }
  }

  const unwatch = (id: string) => {
    watches.delete(id)

    if (watches.size === 0 && timer !== null) {
      window.clearInterval(timer)
      timer = null
    }
  }

  return {
    desktopPluginsRoot: async () => {
      root = (await api<{ root: string }>({ path: '/api/desktop-plugins/root' })).root

      return root
    },
    onPreviewFileChanged: callback => {
      listeners.add(callback)
      const offPreviews = previews.onPreviewFileChanged(callback)

      return () => {
        listeners.delete(callback)
        offPreviews()
      }
    },
    readDir: path => (inside(path) ? get('list', path) : files.readDir(path)),
    readFileText: path => (inside(path) ? readFile(path) : files.readFileText(path)),
    readPluginSource: readFile,
    stopPreviewFileWatch: async id => {
      if (!watches.has(id)) {
        return previews.stopPreviewFileWatch(id)
      }

      unwatch(id)

      return true
    },
    watchPreviewFile: async path => {
      if (!inside(path)) {
        return previews.watchPreviewFile(path)
      }

      const stamp = (await stamps())[path]

      // Electron's answer for a file gone at call time: structured, not a rejection.
      if (stamp === undefined) {
        return { error: 'ENOENT', message: 'File does not exist.', ok: false, path }
      }

      const id = `desktop-plugin:${++nextId}`
      watches.set(id, { path, stamp })
      timer ??= window.setInterval(() => void poll(), PLUGIN_WATCH_POLL_MS)

      return { id, path }
    }
  }
}

import type { HermesApiRequest, HermesReadFileTextResult } from '@/global'

type DesktopPluginsBridge = Pick<
  Window['hermesDesktop'],
  'desktopPluginsRoot' | 'readDir' | 'readFileText' | 'readPluginSource' | 'reconcileDesktopPlugins'
>

interface BrowserDesktopPluginsOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  /** The page's /api/fs readers, kept for every path outside the plugins folder. */
  files: Pick<Window['hermesDesktop'], 'readDir' | 'readFileText'>
}

/** The runtime loader's disk door, answered by the server that serves this page: its app-level
 *  desktop-plugins folder (hermes_cli/web_routers/desktop_plugins.py). /api/fs is the wrong
 *  reader there: it follows the profile's SSH workspace and truncates at the preview cap. */
export function createBrowserDesktopPluginsBridge({ api, files }: BrowserDesktopPluginsOptions): DesktopPluginsBridge {
  let root: null | string = null

  const get = <T>(route: string, path: string) =>
    api<T>({ path: `/api/desktop-plugins/${route}?${new URLSearchParams({ path })}` })

  // The server resolves the folder and builds every path it returns from it, so a path is inside
  // exactly when it starts with the root and a separator. The loader asks for the root first.
  const inside = (path: string) =>
    root !== null && (path === root || path.startsWith(`${root}/`) || path.startsWith(`${root}\\`))

  const readFile = (path: string) => get<HermesReadFileTextResult>('read', path)

  return {
    desktopPluginsRoot: async () => {
      root = (await api<{ root: string }>({ path: '/api/desktop-plugins/root' })).root

      return root
    },
    readDir: path => (inside(path) ? get('list', path) : files.readDir(path)),
    readFileText: path => (inside(path) ? readFile(path) : files.readFileText(path)),
    readPluginSource: readFile,
    // The server copies agent+desktop packages' desktop halves into the folder, as Electron does.
    reconcileDesktopPlugins: async () =>
      (await api<{ touched: string[] }>({ method: 'POST', path: '/api/desktop-plugins/reconcile' })).touched
  }
}

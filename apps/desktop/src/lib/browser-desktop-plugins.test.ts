import { afterEach, describe, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

// The server half (hermes_cli/web_routers/desktop_plugins.py) is pinned in
// tests/hermes_cli/test_webapp_desktop_plugins.py; this is the server's answer for one plugin.
const ROOT = 'C:\\Users\\me\\.hermes\\desktop-plugins'
const FOLDER = `${ROOT}\\notes`

const SERVER: Record<string, unknown> = {
  '/api/desktop-plugins/root': { root: ROOT },
  [`/api/desktop-plugins/list?path=${encodeURIComponent(ROOT)}`]: {
    entries: [{ isDirectory: true, name: 'notes', path: FOLDER }]
  },
  [`/api/desktop-plugins/list?path=${encodeURIComponent(FOLDER)}`]: {
    entries: [{ isDirectory: false, name: 'plugin.js', path: `${FOLDER}\\plugin.js` }]
  },
  [`/api/desktop-plugins/read?path=${encodeURIComponent(`${FOLDER}\\plugin.js`)}`]: {
    path: `${FOLDER}\\plugin.js`,
    text: 'export default { id: "notes", register: globalThis.__notesRegister }',
    truncated: false
  }
}

afterEach(() => {
  clearWebappSession()
  Reflect.deleteProperty(window, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  delete (globalThis as { __notesRegister?: unknown }).__notesRegister
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('browser-hosted Desktop plugins', () => {
  it("loads a plugin from the serving host's desktop-plugins folder, never through /api/fs", async () => {
    seedWebappSession('served-token')
    const requested: string[] = []

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(String(input), window.location.href)
        const route = `${url.pathname}${url.search}`
        requested.push(route)

        return route in SERVER
          ? new Response(JSON.stringify(SERVER[route]), { status: 200 })
          : new Response('Not Found', { status: 404 })
      })
    )

    const register = vi.fn()

    ;(globalThis as { __notesRegister?: unknown }).__notesRegister = register

    // The loader evaluates plugins via blob-URL import(), which vite's module runner can't resolve
    // in tests; a data: URL goes through node's own ESM loader (as in runtime-loader.test.ts).
    vi.spyOn(URL, 'createObjectURL').mockImplementation(
      blob =>
        `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
    )
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    vi.stubGlobal(
      'Blob',
      class {
        parts: string[]
        constructor(parts: string[]) {
          this.parts = parts
        }
      }
    )

    expect(installBrowserDesktopBridge()).toBe(true)
    await discoverRuntimePlugins()

    expect($pluginRecords.get().notes).toMatchObject({ file: `${FOLDER}\\plugin.js`, kind: 'disk', status: 'loaded' })
    expect(register).toHaveBeenCalledTimes(1)
    expect(requested.filter(route => route.startsWith('/api/fs/'))).toEqual([])
  })
})

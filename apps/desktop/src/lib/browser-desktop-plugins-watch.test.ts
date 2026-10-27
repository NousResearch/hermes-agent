import { afterEach, describe, expect, it, vi } from 'vitest'

import { watchRuntimePlugins } from '@/contrib/runtime-loader'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'
import { PLUGIN_WATCH_POLL_MS } from './browser-desktop-plugins'

// The server half (hermes_cli/web_routers/desktop_plugins.py) is pinned in
// tests/hermes_cli/test_webapp_desktop_plugins.py; this is its answer for one plugin, editable.
const ROOT = '/home/me/.hermes/desktop-plugins'
const FOLDER = `${ROOT}/notes`
const ENTRY = `${FOLDER}/plugin.js`

const disk = { source: '', stamp: '' }

function save(version: number) {
  disk.source = `// v${version}\nexport default { id: "notes", register: ctx => globalThis.__notesRegister(${version}) }`
  disk.stamp = `${version}:${disk.source.length}`
}

function answer(route: string): unknown {
  const routes: Record<string, unknown> = {
    '/api/desktop-plugins/root': { root: ROOT },
    '/api/desktop-plugins/stamps': { stamps: { [ENTRY]: disk.stamp } },
    [`/api/desktop-plugins/list?path=${encodeURIComponent(ROOT)}`]: {
      entries: [{ isDirectory: true, name: 'notes', path: FOLDER }]
    },
    [`/api/desktop-plugins/list?path=${encodeURIComponent(FOLDER)}`]: {
      entries: [{ isDirectory: false, name: 'plugin.js', path: ENTRY }]
    },
    [`/api/desktop-plugins/read?path=${encodeURIComponent(ENTRY)}`]: {
      path: ENTRY,
      text: disk.source,
      truncated: false
    }
  }

  return routes[route]
}

afterEach(() => {
  clearWebappSession()
  Reflect.deleteProperty(window, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  delete (globalThis as { __notesRegister?: unknown }).__notesRegister
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('browser-hosted Desktop plugin hot reload', () => {
  it('reloads a plugin when its plugin.js changes on the serving host, as native Desktop does', async () => {
    seedWebappSession('served-token')
    save(1)

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(String(input), window.location.href)
        const body = answer(`${url.pathname}${url.search}`)

        return body === undefined
          ? new Response('Not Found', { status: 404 })
          : new Response(JSON.stringify(body), { status: 200 })
      })
    )

    const register = vi.fn()

    ;(globalThis as { __notesRegister?: unknown }).__notesRegister = register

    // Blob-URL import() can't resolve under vite's module runner; a data: URL goes through node's
    // own ESM loader (as in runtime-loader.test.ts).
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
    watchRuntimePlugins()
    await vi.waitFor(() => expect(register).toHaveBeenCalledWith(1))

    save(2)

    await vi.waitFor(() => expect(register).toHaveBeenLastCalledWith(2), { timeout: PLUGIN_WATCH_POLL_MS * 4 })
    expect(register).toHaveBeenCalledTimes(2)
  })
})

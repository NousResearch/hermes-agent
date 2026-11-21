import { afterEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { installBrowserDesktopBridge } from './browser-desktop-bridge'

type ServedWindow = Window & { __HERMES_DASHBOARD_PROFILE__?: string; hermesDesktop?: Window['hermesDesktop'] }

const win = window as unknown as ServedWindow

const row = (id: string, profile: string) => ({ id, profile, title: id })

function stubGateway() {
  const fetchMock = vi.fn(async (url: URL) => {
    const path = url.pathname

    const body = path.endsWith('/api/profiles/sessions/sidebar')
      ? {
          recents: { sessions: [row('r1', 'default')] },
          cron: { sessions: [] },
          messaging: { sessions: [row('m1', 'work')] }
        }
      : path.endsWith('/api/profiles/sessions')
        ? { sessions: [row('a', 'default'), row('b', 'work')], total: 2 }
        : path.endsWith('/api/sessions/a')
          ? row('a', 'default')
          : {}

    return new Response(JSON.stringify(body), { status: 200 })
  })

  vi.stubGlobal('fetch', fetchMock)
}

afterEach(() => {
  delete win.__HERMES_DASHBOARD_PROFILE__
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  $connection.set(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('session rows a Webapp tab lists', () => {
  it('name the serving gateway as their owner, as the tab sockets and new chats do', async () => {
    stubGateway()
    win.__HERMES_DASHBOARD_PROFILE__ = 'default'
    seedWebappSession('served-token')
    expect(installBrowserDesktopBridge()).toBe(true)
    const api = win.hermesDesktop!.api

    const all = await api<{ sessions: { connection_id?: string }[] }>({ path: '/api/profiles/sessions?limit=40' })
    expect(all.sessions.map(session => session.connection_id)).toEqual(['local', 'local'])

    const sidebar = await api<Record<string, { sessions: { connection_id?: string }[] }>>({
      path: '/api/profiles/sessions/sidebar?limit=40'
    })

    expect(sidebar.recents.sessions[0].connection_id).toBe('local')
    expect(sidebar.messaging.sessions[0].connection_id).toBe('local')
    await expect(api({ path: '/api/sessions/a' })).resolves.toMatchObject({ connection_id: 'local' })
    // A write's echo is not a listing: Electron's main process leaves those untouched too.
    await expect(api({ path: '/api/sessions/a', method: 'PATCH', body: {} })).resolves.not.toHaveProperty(
      'connection_id'
    )
  })
})

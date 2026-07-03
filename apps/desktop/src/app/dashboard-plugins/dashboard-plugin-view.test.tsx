import { cleanup, render, screen, waitFor } from '@testing-library/react'
import type * as React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TRANSLATIONS } from '../../i18n/catalog'

import { resetDashboardPluginRegistryForTests } from './registry'
import {
  dashboardPluginBuildWsUrl,
  dashboardPluginFetchJSON,
  dashboardPluginRequestPath
} from './sdk'

const api = vi.fn()
const getConnection = vi.fn()

function installDesktopBridge() {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api,
      getConnection
    }
  })
}

beforeEach(() => {
  installDesktopBridge()
  api.mockImplementation(async request => {
    if (request.path === '/api/dashboard/plugins') {
      return [
        {
          css: 'dist/style.css',
          entry: 'dist/index.js',
          label: 'Kanban',
          name: 'kanban',
          tab: { path: '/kanban', position: 'after:skills' }
        }
      ]
    }

    if (request.path === '/api/auth/ws-ticket') {
      return { ticket: 'ticket-123' }
    }

    return { ok: true, request }
  })
  getConnection.mockResolvedValue({
    authMode: 'token',
    baseUrl: 'http://127.0.0.1:62659',
    token: 'session-token'
  })
  resetDashboardPluginRegistryForTests()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  resetDashboardPluginRegistryForTests()
  document.querySelectorAll('[data-hermes-dashboard-plugin], [data-hermes-dashboard-plugin-css]').forEach(node =>
    node.remove()
  )
})

describe('desktop dashboard plugin host', () => {
  it('routes plugin JSON calls through the Desktop API bridge', async () => {
    const result = await dashboardPluginFetchJSON('/api/plugins/kanban/tasks', {
      body: JSON.stringify({ title: 'Task' }),
      method: 'POST'
    })

    expect(result).toEqual({
      ok: true,
      request: {
        body: { title: 'Task' },
        method: 'POST',
        path: '/api/plugins/kanban/tasks'
      }
    })
    expect(api).toHaveBeenCalledWith({
      body: { title: 'Task' },
      method: 'POST',
      path: '/api/plugins/kanban/tasks'
    })
  })

  it('normalizes absolute backend URLs back to dashboard-relative API paths', () => {
    expect(dashboardPluginRequestPath('http://127.0.0.1:62659/api/plugins/kanban/board?tenant=eng')).toBe(
      '/api/plugins/kanban/board?tenant=eng'
    )
  })

  it('builds authenticated websocket URLs from the active desktop connection', async () => {
    const url = await dashboardPluginBuildWsUrl('/api/plugins/kanban/events', { board: 'default' })

    expect(url).toBe('ws://127.0.0.1:62659/api/plugins/kanban/events?board=default&token=session-token')
  })

  it('ships Kanban translations to the Desktop plugin SDK catalog', () => {
    expect((TRANSLATIONS.zh.kanban as { columnLabels: Record<string, string> }).columnLabels.todo).toBe('待办')
    expect((TRANSLATIONS.zh.kanban as { orchestration: Record<string, string> }).orchestration.settings).toBe('编排设置')
    expect((TRANSLATIONS.zh.kanban as { orchestration: Record<string, string> }).orchestration.mode).toBe('编排模式')
    expect((TRANSLATIONS['zh-hant'].kanban as { columnLabels: Record<string, string> }).columnLabels.todo).toBe('待辦')
    expect((TRANSLATIONS['zh-hant'].kanban as { orchestration: Record<string, string> }).orchestration.settings).toBe('編排設定')
    expect((TRANSLATIONS.ja.kanban as { columnLabels: Record<string, string> }).columnLabels.done).toBe('完了')
    expect((TRANSLATIONS.ja.kanban as { orchestration: Record<string, string> }).orchestration.settings).toBe(
      'オーケストレーション設定'
    )
  })

  it('loads the shared dashboard plugin bundle and renders the registered component', async () => {
    const originalAppendChild = document.body.appendChild.bind(document.body)
    vi.spyOn(document.body, 'appendChild').mockImplementation((node: Node) => {
      const result = originalAppendChild(node)

      if (node instanceof HTMLScriptElement) {
        queueMicrotask(() => {
          window.__HERMES_PLUGINS__?.register('kanban', (() => (
            <div>Kanban plugin mounted</div>
          )) as React.ComponentType)
          node.onload?.(new Event('load'))
        })
      }

      return result
    })

    const { DashboardPluginView } = await import('./dashboard-plugin-view')
    render(<DashboardPluginView name="kanban" />)

    expect(await screen.findByText('Kanban plugin mounted')).toBeTruthy()
    await waitFor(() => expect(getConnection).toHaveBeenCalled())
    expect(document.querySelector('[data-hermes-dashboard-plugin-host="kanban"]')).toBeTruthy()
    expect(document.body.textContent).toContain('[data-hermes-dashboard-plugin-host] .hermes-kanban-drawer-shade')
    expect(document.body.textContent).toContain('position: absolute')
    expect(document.querySelector('link[data-hermes-dashboard-plugin-css="kanban"]')?.getAttribute('href')).toBe(
      'http://127.0.0.1:62659/dashboard-plugins/kanban/dist/style.css'
    )
    expect(document.querySelector('script[data-hermes-dashboard-plugin="kanban"]')?.getAttribute('src')).toContain(
      'http://127.0.0.1:62659/dashboard-plugins/kanban/dist/index.js'
    )
  })
})

import type { GatewayEvent } from '@hermes/shared'
import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearBrowserTabs, createBrowserTab, setBrowserEnabled, updateBrowserTab } from '@/store/browser'
import { $gateway } from '@/store/gateway'

import { handleBrowserCommandRequest, useBrowserCommandBridge } from './use-browser-command-bridge'

interface Payload {
  command?: unknown
  params?: unknown
  request_id?: unknown
  tab_id?: unknown
}

describe('handleBrowserCommandRequest', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearBrowserTabs()
    setBrowserEnabled(true)
  })

  afterEach(() => {
    cleanup()
    $gateway.set(null)
    clearBrowserTabs()
    setBrowserEnabled(false)
    window.localStorage.clear()
  })

  it('executes a command against the bound visible browser tab and responds through the gateway', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })
    updateBrowserTab(tab.id, { controlMode: 'observe' })
    const requestGateway = vi.fn().mockResolvedValue({ ok: true })
    const runCommand = vi.fn().mockResolvedValue({ title: 'Example' })

    const event: GatewayEvent<Payload> = {
      payload: { command: 'getState', params: { probe: true }, request_id: 'req-1' },
      session_id: 'session-1',
      type: 'browser.command.request'
    }

    await handleBrowserCommandRequest(event, requestGateway, runCommand)

    expect(runCommand).toHaveBeenCalledWith(tab.id, 'getState', { probe: true })
    expect(requestGateway).toHaveBeenCalledWith(
      'browser.desktop.respond',
      { ok: true, request_id: 'req-1', result: { title: 'Example' } },
      10_000
    )
  })

  it('fails closed when no visible tab is bound for the session', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ ok: true })
    const runCommand = vi.fn()

    const event: GatewayEvent<Payload> = {
      payload: { command: 'getState', request_id: 'req-2' },
      session_id: 'session-missing',
      type: 'browser.command.request'
    }

    await handleBrowserCommandRequest(event, requestGateway, runCommand)

    expect(runCommand).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith(
      'browser.desktop.respond',
      {
        error: 'No visible browser tab is bound for this session',
        ok: false,
        request_id: 'req-2'
      },
      10_000
    )
  })

  it('rejects explicit tab_id targeting across sessions before running browser commands', async () => {
    const foreignTab = createBrowserTab({ sessionId: 'session-2', url: 'https://example.com/foreign' })
    updateBrowserTab(foreignTab.id, { controlMode: 'control' })
    const requestGateway = vi.fn().mockResolvedValue({ ok: true })
    const runCommand = vi.fn()

    const event: GatewayEvent<Payload> = {
      payload: { command: 'getState', request_id: 'req-cross-session', tab_id: foreignTab.id },
      session_id: 'session-1',
      type: 'browser.command.request'
    }

    await handleBrowserCommandRequest(event, requestGateway, runCommand)

    expect(runCommand).not.toHaveBeenCalled()
    expect(requestGateway).toHaveBeenCalledWith(
      'browser.desktop.respond',
      {
        error: `Browser tab ${foreignTab.id} belongs to a different session`,
        ok: false,
        request_id: 'req-cross-session'
      },
      10_000
    )
  })

  it('does not subscribe to browser command events when disabled for secondary windows', () => {
    const on = vi.fn()

    $gateway.set({ on } as never)
    renderHook(() => useBrowserCommandBridge(false))

    expect(on).not.toHaveBeenCalled()
  })

  it('ignores malformed events without a request id', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ ok: true })

    const event: GatewayEvent<Payload> = {
      payload: { command: 'getState' },
      session_id: 'session-1',
      type: 'browser.command.request'
    }

    await handleBrowserCommandRequest(event, requestGateway, vi.fn())

    expect(requestGateway).not.toHaveBeenCalled()
  })
})

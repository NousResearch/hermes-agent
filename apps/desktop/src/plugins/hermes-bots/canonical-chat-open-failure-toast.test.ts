/**
 * notifyBotOpenFailure copy contract (desktop-34): every failed bot open
 * toasts a plain-words title + next step from the plugin bundle, keeps the raw
 * RPC/connection error in `detail` only, never says "gateway", and offers the
 * Gateways settings tab as its action.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock, pluginCtx } = vi.hoisted(() => ({
  hostMock: { navigate: vi.fn(), notify: vi.fn(), notifyError: vi.fn(), openSession: vi.fn(), request: vi.fn() },
  pluginCtx: { current: null as null | { i18n?: { t: (key: string, ...args: unknown[]) => string } } }
}))

vi.mock('@hermes/plugin-sdk', () => ({
  BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 15_000,
  host: hostMock,
  usePluginI18n: () => (key: string) => key
}))

vi.mock('./routing', () => ({
  backendTargetProfile: (route: { targetProfile?: string } | null, name: string) => route?.targetProfile ?? name,
  botConnectionRoute: () => null,
  botRosterMeta: () => ({}),
  botWorkspaceOwnerKey: (bot: { name?: string } | null) => `bot:${bot?.name || 'default'}`,
  requestForBot: vi.fn()
}))

vi.mock('./data', () => ({
  $botMeta: { get: () => ({}), set: vi.fn() },
  botMetaKey: (bot: { name?: string }) => bot?.name ?? '',
  botOwner: (owner: RosterRow | string) =>
    typeof owner === 'string'
      ? { bot: { name: owner }, key: owner, name: owner, route: null }
      : { bot: owner, key: owner?.name, name: owner?.name, route: null },
  persistBotMetaSnapshot: vi.fn(),
  saveBotMeta: vi.fn()
}))

vi.mock('./shared', () => ({ getPluginCtx: () => pluginCtx.current }))

type Toast = {
  kind?: string
  title?: string
  message: string
  detail?: string
  action?: { label: string; onClick: () => void }
}

const BOT = { connectionId: 'studio', connectionLabel: 'Studio Mac', name: 'ops' } as RosterRow

const lastToast = (): Toast => hostMock.notify.mock.calls[hostMock.notify.mock.calls.length - 1][0] as Toast

async function load() {
  vi.resetModules()

  return import('./canonical-chat')
}

beforeEach(() => {
  vi.clearAllMocks()
  pluginCtx.current = null
})

describe('notifyBotOpenFailure', () => {
  it('needs-update: names the connection, keeps the raw RPC text in detail, offers Gateways', async () => {
    const { notifyBotOpenFailure } = await load()
    const raw = 'RPC -32601: method not found: bots.canonical'

    notifyBotOpenFailure(new Error(raw), BOT, 'open')

    const toast = lastToast()
    expect(toast.kind).toBe('error')
    expect(toast.title).toBe('This bot lives on an older Hermes')
    expect(toast.message).toBe('Update Studio Mac, then try again.')
    expect(toast.message).not.toContain(raw)
    expect(toast.detail).toBe(raw)
    expect(`${toast.title} ${toast.message}`).not.toMatch(/gateway|backend|worker/i)
    expect(hostMock.notifyError).not.toHaveBeenCalled()

    expect(toast.action?.label).toBe('Open Gateways')
    toast.action?.onClick()
    expect(hostMock.navigate).toHaveBeenCalledWith('/settings?tab=gateway')
  })

  it('unreachable: plain-words title and next step; the connection error is detail only', async () => {
    const { notifyBotOpenFailure } = await load()
    const raw = 'WebSocket connect ECONNREFUSED 10.0.0.7:8642'

    notifyBotOpenFailure(new Error(raw), BOT, 'reach')

    const toast = lastToast()
    expect(toast.kind).toBe('error')
    expect(toast.title).toBe('Hermes couldn’t reach the computer this bot runs on')
    expect(toast.message).toBe('Check it is online and try again.')
    expect(toast.message).not.toContain(raw)
    expect(toast.detail).toBe(raw)
    expect(`${toast.title} ${toast.message}`).not.toMatch(/gateway|backend|worker/i)
    expect(toast.action?.label).toBe('Open Gateways')
    toast.action?.onClick()
    expect(hostMock.navigate).toHaveBeenCalledWith('/settings?tab=gateway')
    expect(hostMock.notifyError).not.toHaveBeenCalled()
  })

  it('open step: names the bot, raw error in detail, same Gateways action', async () => {
    const { notifyBotOpenFailure } = await load()

    notifyBotOpenFailure(new Error('session.list timed out'), BOT, 'open', 'Ops Bot')

    const toast = lastToast()
    expect(toast.title).toBe('Could not open Ops Bot’s chat')
    expect(toast.message).toBe('Try again.')
    expect(toast.detail).toBe('session.list timed out')
    expect(toast.action?.label).toBe('Open Gateways')
  })

  it('resolves copy through the plugin i18n bundle when one is registered', async () => {
    pluginCtx.current = {
      i18n: {
        t: (key: string, ...args: unknown[]) =>
          key === 'bot.openNeedsUpdateMessage' ? `[ja] update ${String(args[0])}` : `[ja] ${key}`
      }
    }
    const { notifyBotOpenFailure } = await load()

    notifyBotOpenFailure(new Error('unknown method'), BOT, 'open')

    expect(lastToast()).toMatchObject({
      title: '[ja] bot.openNeedsUpdateTitle',
      message: '[ja] update Studio Mac',
      action: { label: '[ja] bot.openGateways' }
    })
  })
})

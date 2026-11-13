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
    expect(toast.message).toContain('Studio Mac')
    expect(toast.message).not.toContain(raw)
    expect(toast.detail).toBe(raw)
    expect(`${toast.title} ${toast.message}`).not.toMatch(/gateway|backend|worker/i)
    expect(hostMock.notifyError).not.toHaveBeenCalled()

    toast.action?.onClick()
    expect(hostMock.navigate).toHaveBeenCalledWith('/settings?tab=gateway')
  })

  it('unreachable: plain-words title and next step; the connection error is detail only', async () => {
    const { notifyBotOpenFailure } = await load()
    const raw = 'WebSocket connect ECONNREFUSED 10.0.0.7:8642'

    notifyBotOpenFailure(new Error(raw), BOT, 'reach')

    const toast = lastToast()
    expect(toast.kind).toBe('error')
    expect(toast.message).not.toContain(raw)
    expect(toast.detail).toBe(raw)
    expect(`${toast.title} ${toast.message}`).not.toMatch(/gateway|backend|worker/i)
    toast.action?.onClick()
    expect(hostMock.navigate).toHaveBeenCalledWith('/settings?tab=gateway')
    expect(hostMock.notifyError).not.toHaveBeenCalled()
  })

  it('a redesign agent row: names the agent as the row does and says why in one plain line', async () => {
    const { notifyBotOpenFailure } = await load()
    const raw = 'Could not check research\'s Bot Chat registry (This machine does not serve agent "research", so this tab cannot open it.) — not starting a new chat'

    // The roster calls this profile "Hermes" (a cloned title); the row reads "research".
    notifyBotOpenFailure(new Error(raw), { ...BOT, name: 'research' }, 'open', { botName: 'research', explain: true })

    const toast = lastToast()
    expect(toast.title).toBe('Could not open research’s chat')
    expect(toast.message).toBe('research isn’t running on this machine right now — try again in a moment.')
    expect(toast.detail).toBe(raw)
    expect(`${toast.title} ${toast.message}`).not.toMatch(/gateway|backend|worker|registry|Bot Chat/i)
  })

  it('a redesign agent row that cannot reach its machine keeps the agent in the title', async () => {
    const { notifyBotOpenFailure } = await load()

    notifyBotOpenFailure(new Error('WebSocket connect ECONNREFUSED 10.0.0.7:8642'), BOT, 'reach', {
      botName: 'ops',
      explain: true
    })

    const toast = lastToast()
    expect(toast.title).toBe('Could not open ops’s chat')
    expect(toast.message).toBe('Couldn’t reach the machine it runs on — check that it’s on and connected.')
    expect(toast.message).not.toContain('ECONNREFUSED')
  })

  it('an unrecognised failure keeps the plain next step and the raw text in detail', async () => {
    const { notifyBotOpenFailure } = await load()

    notifyBotOpenFailure(new Error('RPC -32000: session.create exploded'), BOT, 'open', { botName: 'ops', explain: true })

    expect(lastToast()).toMatchObject({ title: 'Could not open ops’s chat', message: 'Try again.', detail: 'RPC -32000: session.create exploded' })
  })

  it('a redesign agent row refused for version skew says to update, not just "try again"', async () => {
    const { notifyBotOpenFailure } = await load()
    const raw = 'invalid params for session.create: request_id: Extra inputs are not permitted — the client and the Hermes backend are out of sync (different versions); run `hermes update` and restart both'

    notifyBotOpenFailure(new Error(raw), BOT, 'open', { botName: 'default', explain: true })

    expect(lastToast()).toMatchObject({ title: 'Could not open default’s chat', message: 'Update Studio Mac, then try again.', detail: raw })
  })
})

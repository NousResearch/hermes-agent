import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { GatewayEventContext } from './types'

const sent: { method: string; params: Record<string, unknown>; url: string }[] = []

class FakeGateway {
  connectionState = 'closed'
  url = 'ws://default'
  async connect(url: string) {
    this.url = url
    this.connectionState = 'open'
  }
  async request(method: string, params: Record<string, unknown> = {}) {
    sent.push({ method, params, url: this.url })

    return { status: 'ok' }
  }
  close() {
    this.connectionState = 'closed'
  }
  onEvent() {
    return () => undefined
  }
  onState() {
    return () => undefined
  }
}
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  HermesGateway: FakeGateway,
  setApiRequestConnection: vi.fn()
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn(), setMessages: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
vi.mock('@/app/chat/right-rail/preview-reader', () => ({
  readActivePreview: vi.fn(async () => ({ text: 'public page' }))
}))
vi.mock('@/app/right-sidebar/terminal/agent-terminal-stream', () => ({ writeAgentTerminalChunk: vi.fn() }))
vi.mock('@/app/right-sidebar/terminal/buffer', () => ({ readActiveTerminal: vi.fn(() => ({ text: 'terminal' })) }))
vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ closeAgentTerminalByProc: vi.fn() }))
vi.mock('@/store/pane-focus', () => ({ applyDesktopLayoutPreset: vi.fn(), revealDesktopPane: vi.fn() }))
vi.mock('@/store/reactions-local', () => ({ recordAgentReaction: vi.fn() }))
vi.mock('@/store/tips', () => ({ $tipsEnabled: { get: () => false }, showTip: vi.fn() }))
vi.mock('@/store/tours', () => ({ $toursEnabled: { get: () => false } }))
const { handleDesktopBridgeEvent } = await import('./desktop-bridge')

const { $gateway, closeSecondaryGateways, configureGatewayRegistry, setPrimaryGateway } =
  await import('@/store/gateway')

function request(type: string, profile = 'juststoreit', connectionId = 'local') {
  const payload = { request_id: 'probe-request', action: 'elements' }

  return {
    event: { type, profile, connectionId, session_id: 'jsi-runtime' },
    payload,
    isActiveEvent: false
  } as GatewayEventContext
}

beforeEach(() => {
  sent.length = 0
  configureGatewayRegistry({ onEvent: vi.fn() })
  const primary = new FakeGateway()
  primary.connectionState = 'open'
  setPrimaryGateway(primary as never, 'default')
  $gateway.set(primary as never)
  Object.assign(window, {
    hermesDesktop: {
      getConnection: async () => ({ mode: 'local' }),
      getConnectionFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
        connectionId,
        profile,
        mode: 'local'
      }),
      getGatewayWsUrlFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        `ws://${connectionId}/${profile}`,
      touchBackend: async () => undefined,
      readWindowBelow: async () => ({ window: { app: 'Hermes' } })
    }
  })
})
afterEach(() => {
  closeSecondaryGateways()
  vi.restoreAllMocks()
})
it.each(['preview.act', 'preview.read', 'window.read', 'terminal.read', 'tour'])(
  'answers %s on the requesting profile socket, even when default is active',
  async kind => {
    expect(handleDesktopBridgeEvent(request(`${kind}.request`))).toBe(true)
    await vi.waitFor(() => expect(sent.some(call => call.method === `${kind}.respond`)).toBe(true))
    expect(sent.find(call => call.method === `${kind}.respond`)).toMatchObject({
      url: 'ws://local/juststoreit',
      params: { request_id: 'probe-request' }
    })
  }
)
it('keeps an asynchronous reply on its original connection when foreground changes', async () => {
  let finish!: (value: null) => void
  vi.spyOn(window.hermesDesktop!, 'readWindowBelow').mockImplementation(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  handleDesktopBridgeEvent(request('window.read.request', 'default', 'source-a'))
  const other = new FakeGateway()
  other.url = 'ws://source-b/default'
  $gateway.set(other as never)
  finish(null)
  await vi.waitFor(() => expect(sent.some(call => call.method === 'window.read.respond')).toBe(true))
  expect(sent.find(call => call.method === 'window.read.respond')?.url).toBe('ws://source-a/default')
})

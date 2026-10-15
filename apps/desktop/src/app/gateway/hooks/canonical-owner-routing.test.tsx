import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { activeGateway, closeSecondaryGateways } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { clearAllSessionStates, runtimeSessionOwner } from '@/store/session-states'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

const sharedPrimaryConn = {
  authMode: 'token' as const, baseUrl: 'http://127.0.0.1:8899', connectionId: '',
  profile: 'default', sharedPrimary: true, token: 'fixture',
  wsUrl: 'ws://127.0.0.1:8899/api/ws?native_dial=fixture&ticket=one-use'
}

const originalWebSocket = globalThis.WebSocket

function fakeDesktop() {
  return {
    getConnection: vi.fn(async () => sharedPrimaryConn),
    getGatewayWsUrl: vi.fn(async () => sharedPrimaryConn.wsUrl),
    getBootProgress: vi.fn(async () => ({ error: null, fakeMode: false, message: '', phase: 'init',
      progress: 0, retryable: false, running: true, timestamp: Date.now() })),
    onBootProgress: vi.fn(() => () => undefined), onBackendExit: vi.fn(() => () => undefined),
    onConnectionApplied: vi.fn(() => () => undefined), onPowerResume: vi.fn(() => () => undefined),
    onWindowStateChanged: vi.fn(() => () => undefined),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

function Harness({ handleGatewayEvent, handleServerRequest }: Pick<Parameters<typeof useGatewayBoot>[0], 'handleGatewayEvent' | 'handleServerRequest'>) {
  useGatewayBoot({ handleGatewayEvent, handleServerRequest, beforeConnectionSwitch: () => undefined,
    onConnectionReady: () => undefined, onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined, refreshSessions: async () => undefined })

  return null
}

beforeEach(() => {
  takeGatewaySurvivor()?.gateway.close()
  closeSecondaryGateways()
  clearAllSessionStates()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  $gatewayState.set('idle')
})

afterEach(() => {
  cleanup()
  takeGatewaySurvivor()?.gateway.close()
  closeSecondaryGateways()
  clearAllSessionStates()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

async function flushAsync() {
  await act(async () => { await vi.advanceTimersByTimeAsync(0) })
}

function deliverEvent(socket: FakeWebSocket, frame: Record<string, unknown>) {
  ;(socket as unknown as { emit: (type: string, ev: unknown) => void }).emit('message', {
    data: JSON.stringify({ jsonrpc: '2.0', method: 'event', params: frame })
  })
}

test('keeps canonical alpha events and questions on alpha while beta is foreground', async () => {
    const desktop = fakeDesktop()

    const native = { ...sharedPrimaryConn, profile: 'default',
      wsUrl: 'ws://127.0.0.1:8899/api/ws?native_dial=fixture&ticket=one-use' }

    desktop.getConnection.mockResolvedValue(native)
    desktop.getGatewayWsUrl.mockResolvedValue(native.wsUrl)
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    const events = vi.fn()
    const questions = vi.fn((_request: unknown) => true)
    render(<Harness handleGatewayEvent={events} handleServerRequest={questions} />)
    await flushAsync()
    const socket = FakeWebSocket.instances[0]

    const emit = (data: object) => (socket as unknown as { emit: (type: string, ev: unknown) => void })
      .emit('message', { data: JSON.stringify(data) })

    const send = vi.spyOn(socket, 'send').mockImplementation(data => {
      const frame = JSON.parse(data)
      emit({ jsonrpc: '2.0', id: frame.id, result: frame.method === 'session.resume'
        ? { session_id: 'rt-alpha', revision: 1, execution_generation: 2, replay_epoch: `${frame.params.profile}-epoch`, prompts: [] }
        : { status: 'resolved' } })
    })

    try {
      await act(async () => { await activeGateway()!.request('session.resume', { session_id: 'rt-alpha', profile: 'alpha' }) })
      await act(async () => { await activeGateway()!.request('session.resume', { session_id: 'rt-alpha', profile: 'beta' }) })
      act(() => { $activeGatewayProfile.set('beta') })
      act(() => {
        deliverEvent(socket, { session_id: 'rt-alpha', type: 'session.info', replay_epoch: 'beta-epoch', seq: 1,
          payload: { revision: 2, execution_generation: 3 } })
        deliverEvent(socket, { session_id: 'rt-alpha', type: 'session.info', replay_epoch: 'alpha-epoch', seq: 1,
          payload: { revision: 2, execution_generation: 3 } })
        deliverEvent(socket, { session_id: 'rt-alpha', type: 'approval.request', replay_epoch: 'alpha-epoch', seq: 2,
          payload: { kind: 'approval', prompt_id: 'alpha-approval', execution_generation: 3, command: 'build', choices: ['once'] } })
      })
      expect(events.mock.calls.at(-1)?.[0]).toMatchObject({ profile: 'alpha', session_id: 'rt-alpha' })
      expect(events.mock.calls.map(([event]) => event.profile)).toEqual(['beta', 'alpha', 'alpha'])
      expect(runtimeSessionOwner('rt-alpha')).toBe('alpha')
      expect(questions.mock.calls.at(-1)?.[0]).toMatchObject({ profile: 'alpha', id: 'alpha-approval' })
      const question = questions.mock.calls.at(-1)?.[0] as unknown as { respond: (answer: object) => void }
      await act(async () => { question.respond({ choice: 'once' }) })
      expect(send.mock.calls.map(([data]) => JSON.parse(data)).find(frame => frame.method === 'approval.respond')?.params.profile).toBe('alpha')
    } finally { send.mockRestore() }
  })

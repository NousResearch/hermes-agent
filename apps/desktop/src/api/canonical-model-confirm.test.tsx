import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { $activeSessionId, $currentModel, setCurrentModel, setCurrentProvider } from '@/store/session'

import { CanonicalDesktopProtocol } from './canonical-protocol'

const confirmMock = vi.fn()

vi.mock('@/store/confirm', () => ({ confirm: (...args: unknown[]) => confirmMock(...args) }))
vi.mock('@/store/notifications', () => ({ dismissNotification: vi.fn(), notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/hermes', () => ({ getGlobalModelInfo: vi.fn(), setApiRequestProfile: vi.fn(), setGlobalModel: vi.fn() }))

// The canonical owner: a guarded target answers `confirmation_required` + a one-time token and
// writes nothing; only that token in `payload.confirm` applies it.
function owner() {
  const state = { model: 'old', revision: 3, generation: 2 }
  const mutations: Record<string, unknown>[] = []
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 3, execution_generation: 2 })

  const request = async <T,>(method: string, params: Record<string, unknown> = {}): Promise<T> => {
    const prepared = protocol.prepare(method, params)
    expect(protocol.wire(method, prepared)).toBe('session.mutate')
    const payload = prepared.payload as Record<string, unknown>
    mutations.push(payload)

    const value = payload.confirm === 'tok-1'
      ? (Object.assign(state, { model: payload.model, revision: 4, generation: 3 }),
        { session_id: 's', operation: 'model', revision: 4, execution_generation: 3, model: payload.model })
      : { session_id: 's', operation: 'model', status: 'confirmation_required', confirm: 'tok-1',
          confirm_message: 'This session holds ~5,000 tokens.', target_model: payload.model }

    return protocol.result(method, prepared, value) as T
  }

  return { mutations, protocol, request, state }
}

function Picker({ onReady, request }: { onReady: (c: ReturnType<typeof useModelControls>) => void; request: ReturnType<typeof owner>['request'] }) {
  onReady(useModelControls({ queryClient: new QueryClient(), requestGateway: request }))

  return null
}

afterEach(() => {
  cleanup()
  confirmMock.mockReset()
  $activeSessionId.set(null)
})

test('a guarded canonical picker switch asks; yes applies once with the owner token, no changes nothing', async () => {
  for (const accept of [false, true]) {
    const gateway = owner()
    $activeSessionId.set('s')
    setCurrentModel('old')
    setCurrentProvider('custom')
    confirmMock.mockResolvedValueOnce(accept)
    let controls!: ReturnType<typeof useModelControls>
    render(<Picker onReady={value => (controls = value)} request={gateway.request} />)

    await act(async () => {
      await controls.selectModel({ model: 'pricey', provider: 'custom' })
    })
    await waitFor(() => expect(confirmMock).toHaveBeenCalledWith(expect.objectContaining({
      description: 'This session holds ~5,000 tokens.' })))
    await act(async () => {})

    expect(gateway.mutations).toEqual(accept
      ? [{ model: 'pricey', provider: 'custom' }, { model: 'pricey', provider: 'custom', confirm: 'tok-1' }]
      : [{ model: 'pricey', provider: 'custom' }])
    expect(gateway.state.model).toBe(accept ? 'pricey' : 'old')
    expect($currentModel.get()).toBe(accept ? 'pricey' : 'old')
    cleanup()
    confirmMock.mockReset()
  }
})

test('a typed /model on a canonical session answers the same handshake and its resend carries the token once', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 3, execution_generation: 2 })
  const typed = protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey' })

  const refused = protocol.result('slash.exec', typed, { session_id: 's', operation: 'model', status: 'confirmation_required',
    confirm: 'tok-1', confirm_message: 'pricey IS EXPENSIVE' })

  expect(refused).toMatchObject({ confirm_required: true, confirm_message: 'pricey IS EXPENSIVE' })
  expect(refused).not.toHaveProperty('model')
  const resend = protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey', confirm_expensive_model: true })
  expect(resend.payload).toEqual({ model: 'pricey', confirm: 'tok-1' })
  expect(resend.request_id).not.toBe(typed.request_id)
  // Consumed: a second refusal is never auto-confirmed with the spent token.
  protocol.result('slash.exec', resend, { session_id: 's', operation: 'model', status: 'confirmation_required', confirm: 'tok-2', confirm_message: 'again' })
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 3, execution_generation: 2 })
  expect(protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey' }).payload).toEqual({ model: 'pricey' })
})

test('a confirmed model switch whose reply was lost retries as the exact token-bearing request until answered', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 3, execution_generation: 2 })
  const pick = { session_id: 's', key: 'model', value: 'pricey --provider custom' }
  const refused = protocol.prepare('config.set', pick)
  protocol.result('config.set', refused, { session_id: 's', operation: 'model', status: 'confirmation_required', confirm: 'tok-1', confirm_message: 'pricey' })

  // The user confirmed; the owner may have committed it, but the reply never arrived.
  const confirmed = protocol.prepare('config.set', { ...pick, confirm_expensive_model: true })
  expect(confirmed.payload).toEqual({ model: 'pricey', provider: 'custom', confirm: 'tok-1' })
  protocol.failure(confirmed, new Error('Hermes gateway connection closed'))

  // Every retry of that target is the same confirmed intent: same request id, same token.
  for (const retry of [{ ...pick, confirm_expensive_model: true }, pick]) {
    const again = protocol.prepare('config.set', retry)
    expect(again.request_id).toBe(confirmed.request_id)
    expect(again.payload).toEqual(confirmed.payload)
  }

  // A different target never inherits the token.
  expect(protocol.prepare('config.set', { ...pick, value: 'other --provider custom', confirm_expensive_model: true }).payload)
    .toEqual({ model: 'other', provider: 'custom' })

  // The owner's answer spends it: the next switch to the same target asks afresh.
  protocol.result('config.set', confirmed, { session_id: 's', operation: 'model', revision: 4, execution_generation: 3, model: 'pricey' })
  expect(protocol.prepare('config.set', { ...pick, confirm_expensive_model: true }).payload).toEqual({ model: 'pricey', provider: 'custom' })
})

test('a typed refusal of the confirmed request spends its token; a declined dialog never sends one', () => {
  const protocol = new CanonicalDesktopProtocol()
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 3, execution_generation: 2 })

  const refuse = (prepared: Record<string, unknown>, token: string) =>
    protocol.result('slash.exec', prepared, { session_id: 's', operation: 'model', status: 'confirmation_required', confirm: token, confirm_message: 'x' })

  refuse(protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey' }), 'tok-1')
  // Declined: the user's next plain attempt of the same target carries no token.
  expect(protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey' }).payload).toEqual({ model: 'pricey' })

  const confirmed = protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey', confirm_expensive_model: true })
  expect(confirmed.payload).toEqual({ model: 'pricey', confirm: 'tok-1' })
  const stale = Object.assign(new Error('stale_generation'), { data: { reason: 'stale_generation' } })
  protocol.failure(confirmed, stale)
  protocol.result('session.resume', { session_id: 's' }, { session_id: 's', revision: 3, execution_generation: 2 })
  expect(protocol.prepare('slash.exec', { session_id: 's', command: 'model pricey', confirm_expensive_model: true }).payload).toEqual({ model: 'pricey' })
})

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { MemoryConnect } from './connect'

const api = vi.hoisted(() => ({ status: vi.fn(), start: vi.fn() }))
vi.mock('@/hermes', () => ({ getMemoryProviderOAuthStatus: api.status, startMemoryProviderOAuth: api.start }))

const idle = { state: 'idle' as const, connected: false, auth: null, detail: '' }
const pending = { ...idle, state: 'pending' as const }
const connected = { ...idle, state: 'connected' as const, connected: true, auth: 'oauth' as const }
const owner = { connectionId: 'remote-a', profile: 'alpha' }
const other = { ...owner, connectionId: 'remote-b' }
const settle = () => act(async () => {})
const advance = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms))

async function click(name: string) {
  await settle()
  fireEvent.click(screen.getByRole('button', { name }))
  await settle()
}

beforeEach(() => {
  vi.useFakeTimers()
  api.status.mockReset().mockResolvedValue(idle)
  api.start.mockReset().mockResolvedValue(pending)
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('Connect starts and polls for the owner; Stop waiting ends polling without cancelling; connected fires onConnected; another owner’s late response never renders; supported:false hides Connect', async () => {
  const onConnected = vi.fn()
  api.status.mockResolvedValueOnce(idle).mockResolvedValue(pending)
  const view = render(<MemoryConnect onConnected={onConnected} owner={owner} provider="one" />)
  await click('Connect')
  await advance(1500)
  expect(api.start).toHaveBeenCalledExactlyOnceWith('one', owner)
  expect(api.status).toHaveBeenLastCalledWith('one', owner)
  const polls = api.status.mock.calls.length
  await click('Stop waiting')
  await advance(10_000)
  expect(api.status).toHaveBeenCalledTimes(polls)
  api.status.mockResolvedValue(connected)
  await click('Retry connection check')
  expect(screen.getByText('OAuth connected')).toBeTruthy()
  expect(onConnected).toHaveBeenCalledTimes(1)
  expect(api.start).toHaveBeenCalledTimes(1)

  let late!: (value: typeof connected) => void
  api.status
    .mockReset()
    .mockImplementationOnce(() => new Promise(resolve => (late = resolve)))
    .mockResolvedValue(idle)
  view.rerender(<MemoryConnect owner={other} provider="one" />)
  await settle()
  view.rerender(<MemoryConnect owner={owner} provider="one" />)
  await settle()
  late(connected)
  await settle()
  expect(api.status).toHaveBeenNthCalledWith(1, 'one', other)
  expect(screen.queryByText('OAuth connected')).toBeNull()
  expect(screen.getByRole('button', { name: 'Connect' })).toBeTruthy()

  api.status.mockResolvedValue({ ...idle, supported: false })
  view.rerender(<MemoryConnect owner={owner} provider="two" />)
  await settle()
  expect(view.container.textContent).toBe('')
})

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { MemoryConnect } from './connect'

const api = vi.hoisted(() => ({ status: vi.fn(), start: vi.fn() }))
vi.mock('@/hermes', () => ({ getMemoryProviderOAuthStatus: api.status, startMemoryProviderOAuth: api.start }))

const idle = { state: 'idle' as const, connected: false, auth: null, detail: '' }
const pending = { ...idle, state: 'pending' as const }
const connected = { ...idle, state: 'connected' as const, connected: true, auth: 'oauth' as const }
const owner = { connectionId: 'remote-a', profile: 'alpha' }
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

it('Connect starts the flow for the owner and polls; Stop waiting ends polling without cancelling; Retry resumes', async () => {
  api.status.mockResolvedValueOnce(idle).mockResolvedValue(pending)
  render(<MemoryConnect profile={owner} provider="one" />)
  await click('Connect')
  await advance(1500)
  expect(api.start).toHaveBeenCalledExactlyOnceWith('one', owner)
  const polls = api.status.mock.calls.length
  await click('Stop waiting')
  await advance(10_000)
  expect(api.status).toHaveBeenCalledTimes(polls)
  api.status.mockResolvedValue(connected)
  await click('Retry connection check')
  expect(screen.getByText('OAuth connected')).toBeTruthy()
  expect(api.start).toHaveBeenCalledTimes(1)
})

it('never renders one owner’s response on another owner’s view', async () => {
  let late!: (value: typeof connected) => void
  api.status.mockResolvedValueOnce(idle).mockImplementationOnce(() => new Promise(resolve => (late = resolve)))
  const view = render(<MemoryConnect profile={owner} provider="one" />)
  await click('Connect')
  await advance(1500)
  view.rerender(<MemoryConnect profile={{ ...owner, connectionId: 'remote-b' }} provider="one" />)
  await settle()
  late(connected)
  await settle()
  expect(screen.queryByText('OAuth connected')).toBeNull()
  expect(api.status).toHaveBeenLastCalledWith('one', { ...owner, connectionId: 'remote-b' })
})

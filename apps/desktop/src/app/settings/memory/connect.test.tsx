import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MemoryProviderOAuthStatus } from '@/types/hermes'

const { getStatus, start, notifyError } = vi.hoisted(() => ({
  getStatus: vi.fn(),
  start: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/hermes', () => ({ getMemoryProviderOAuthStatus: getStatus, startMemoryProviderOAuth: start }))
vi.mock('@/store/notifications', () => ({ notifyError }))
const { MemoryConnect } = await import('./connect')

const idle = { connected: false, auth: null, state: 'idle', detail: '' } as MemoryProviderOAuthStatus
const connected = { ...idle, connected: true, auth: 'oauth' } as MemoryProviderOAuthStatus
const owner = { connectionId: 'remote-a', profile: 'a' }

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

beforeEach(() => {
  vi.useFakeTimers()
  getStatus.mockResolvedValue(idle)
  start.mockResolvedValue({ ...idle, state: 'pending' })
})
afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.resetAllMocks()
})

describe('MemoryConnect', () => {
  describe.each(['start', 'poll'] as const)('pending %s', phase => {
    it.each(['stop', 'unmount', 'owner', 'deadline'] as const)('retires on %s without late effects', async action => {
      const pending = deferred<MemoryProviderOAuthStatus>()
      const onConnected = vi.fn().mockResolvedValue(undefined)

      if (phase === 'start') {
        start.mockReturnValue(pending.promise)
      }

      const result = render(<MemoryConnect onConnected={onConnected} owner={owner} provider="honcho" />)
      await act(async () => {})
      await act(async () => {
        const button = screen.getByRole('button', { name: 'Connect' })
        fireEvent.click(button)
        // Same-batch clicks must not admit another start, even while hung.
        fireEvent.click(button)
      })
      expect(start).toHaveBeenCalledExactlyOnceWith('honcho', undefined, owner)

      if (phase === 'poll') {
        getStatus.mockReturnValueOnce(pending.promise)
        await act(async () => {
          await vi.advanceTimersByTimeAsync(1500)
        })
      }

      expect(getStatus).toHaveBeenCalledTimes(phase === 'start' ? 1 : 2)

      const retire = {
        stop: () => fireEvent.click(screen.getByRole('button', { name: 'Stop waiting' })),
        unmount: () => result.unmount(),
        owner: () =>
          result.rerender(
            <MemoryConnect
              onConnected={onConnected}
              owner={{ connectionId: phase === 'start' ? 'remote-b' : null, profile: 'b' }}
              provider="honcho"
            />
          ),
        deadline: () => vi.advanceTimersByTimeAsync(phase === 'start' ? 120_000 : 118_500)
      }

      await act(async () => {
        await retire[action]()
      })
      expect(screen.queryByText('Waiting for browser consent…')).toBeNull()

      if (action === 'deadline') {
        expect(getStatus).toHaveBeenCalledTimes(phase === 'start' ? 1 : 2)
      }

      const probes = getStatus.mock.calls.length
      getStatus.mockResolvedValue(connected)
      await act(async () => {
        pending.resolve(phase === 'start' ? idle : connected)
        await vi.advanceTimersByTimeAsync(5000)
      })
      expect(getStatus).toHaveBeenCalledTimes(probes)
      expect(onConnected).not.toHaveBeenCalled()
      expect(screen.queryByText('oauth set')).toBeNull()
    })
  })
})

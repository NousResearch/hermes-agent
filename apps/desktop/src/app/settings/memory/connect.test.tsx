import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import { MemoryConnect } from './connect'

const owner = { connectionId: 'source-a', profile: 'default' }
const idle = { connected: false, state: 'idle', auth: null }

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

it('a delayed OAuth start cannot create polling after unmount or cancellation', async () => {
  vi.useFakeTimers()

  for (const unmount of [true, false]) {
    let resolve!: (value: unknown) => void

    const start = new Promise(yes => {
      resolve = yes
    })

    const api = vi.fn((request: HermesApiRequest) => (request.path.endsWith('/start') ? start : Promise.resolve(idle)))
    vi.stubGlobal('hermesDesktop', { api })
    const view = render(<MemoryConnect profile={owner} provider="fixture" />)
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    if (unmount) {
      view.unmount()
    } else {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    }

    await act(async () => {
      resolve(idle)
      await vi.advanceTimersByTimeAsync(130_000)
    })
    expect(api).toHaveBeenCalledTimes(2)
    expect(
      api.mock.calls.every(
        ([request]) => request.connectionId === owner.connectionId && request.profile === owner.profile
      )
    ).toBe(true)
    view.unmount()
  }
})

it('the OAuth deadline expires even when every poll rejects or never resolves', async () => {
  vi.useFakeTimers()

  for (const hangs of [false, true]) {
    const api = vi.fn((request: HermesApiRequest) => {
      if (request.path.endsWith('/start') || api.mock.calls.length === 1) {
        return Promise.resolve(idle)
      }

      return hangs ? new Promise(() => {}) : Promise.reject(new Error('offline'))
    })

    vi.stubGlobal('hermesDesktop', { api })
    const view = render(<MemoryConnect profile={owner} provider="fixture" />)
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000)
    })
    expect(screen.getByText('Timed out — try again.')).toBeTruthy()
    const calls = api.mock.calls.length
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect(api).toHaveBeenCalledTimes(calls)
    view.unmount()
  }
})

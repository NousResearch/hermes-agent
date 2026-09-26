import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { NvidiaEglFallbackBanner } from './nvidia-egl-fallback-banner'

const notify = vi.hoisted(() => vi.fn())

vi.mock('@/store/notifications', () => ({ notify }))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  notify.mockClear()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it('surfaces a persistent notice when the main process reports an active NVIDIA EGL fallback', async () => {
  const getNvidiaEglFallbackReason = vi.fn().mockResolvedValue('NVIDIA driver 580 (>= 580)')
  window.hermesDesktop = { getNvidiaEglFallbackReason } as never

  render(<NvidiaEglFallbackBanner />)

  await waitFor(() => expect(notify).toHaveBeenCalledTimes(1))
  const [toast] = notify.mock.calls[0]
  expect(toast.durationMs).toBe(0)
  expect(toast.message).toContain('NVIDIA driver 580 (>= 580)')
})

it('stays silent when the main process reports no fallback', async () => {
  const getNvidiaEglFallbackReason = vi.fn().mockResolvedValue(null)
  window.hermesDesktop = { getNvidiaEglFallbackReason } as never

  render(<NvidiaEglFallbackBanner />)

  await waitFor(() => expect(getNvidiaEglFallbackReason).toHaveBeenCalled())
  expect(notify).not.toHaveBeenCalled()
})

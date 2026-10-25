import { describe, expect, it, vi } from 'vitest'

import { reportStartupLatency } from './report-startup-latency'

describe('reportStartupLatency', () => {
  it('sends the metric once even when the renderer boots again (reload, reconnect, extra window)', async () => {
    // The main-process latch: a number for the first claim of the launch, null afterwards.
    const claimStartupLatency = vi.fn().mockResolvedValueOnce(1234).mockResolvedValue(null)

    // Older backends reject the unknown method; the report must swallow it.
    const request = vi.fn().mockRejectedValue(new Error('method not found'))

    await reportStartupLatency({ claimStartupLatency }, request)
    await reportStartupLatency({ claimStartupLatency }, request)
    await reportStartupLatency({ claimStartupLatency }, request)

    expect(request).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith('shared_metrics.startup_latency', {
      elapsed_ms: 1234,
      launch_id: expect.any(String),
      surface: 'desktop_attach'
    })
  })

  it('skips the metric on an Electron shell without the claim bridge', async () => {
    const request = vi.fn()

    await reportStartupLatency({}, request)
    await reportStartupLatency(undefined, request)

    expect(request).not.toHaveBeenCalled()
  })
})

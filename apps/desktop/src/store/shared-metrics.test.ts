import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetOptionalIpcLatchesForTests } from '@/lib/optional-ipc'

import { reportPendingUpdateRun } from './shared-metrics'

const noHandler = new Error(
  "Error occurred in handler for 'hermes:updates:metric:take': Error: No handler registered for 'hermes:updates:metric:take'"
)

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

describe('reportPendingUpdateRun against an older main process', () => {
  let takePendingRun: ReturnType<typeof vi.fn>
  let ackPendingRun: ReturnType<typeof vi.fn>

  beforeEach(() => {
    resetOptionalIpcLatchesForTests()
    takePendingRun = vi.fn()
    ackPendingRun = vi.fn(async () => undefined)
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      updates: { ackPendingRun, takePendingRun }
    }
  })

  it('a No handler registered rejection latches the channel for the session', async () => {
    takePendingRun.mockRejectedValue(noHandler)

    const request = vi.fn(async () => ({}) as never)

    await reportPendingUpdateRun(request)
    await reportPendingUpdateRun(request)
    await reportPendingUpdateRun(request)

    expect(takePendingRun).toHaveBeenCalledTimes(1)
    expect(request).not.toHaveBeenCalled()
    expect(ackPendingRun).not.toHaveBeenCalled()
  })

  it('any other rejection keeps the channel alive for the next attach', async () => {
    takePendingRun.mockRejectedValue(new Error('the main process shut down mid-reload'))

    const request = vi.fn(async () => ({}) as never)

    await reportPendingUpdateRun(request)
    await reportPendingUpdateRun(request)

    expect(takePendingRun).toHaveBeenCalledTimes(2)
  })

  it('a latched channel is skipped without touching the bridge at all', async () => {
    takePendingRun.mockRejectedValueOnce(noHandler)

    const request = vi.fn(async () => ({}) as never)

    await reportPendingUpdateRun(request)
    takePendingRun.mockClear()
    takePendingRun.mockResolvedValue({
      duration_ms: 1,
      mechanism: 'posix-handoff',
      outcome: 'success'
    })

    await reportPendingUpdateRun(request)

    expect(takePendingRun).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })
})

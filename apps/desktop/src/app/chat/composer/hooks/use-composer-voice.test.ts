import { describe, expect, it, vi } from 'vitest'

import { createWakePauseCoordinator } from './use-composer-voice'

describe('createWakePauseCoordinator', () => {
  it('waits for a pending pause before resuming after cancellation', async () => {
    let resolvePause: (() => void) | undefined

    const pause = vi.fn(
      () =>
        new Promise<void>(resolve => {
          resolvePause = resolve
        })
    )

    const resume = vi.fn().mockResolvedValue(undefined)
    const coordinator = createWakePauseCoordinator({ pause, resume })

    coordinator.pause()
    const resuming = coordinator.resume()
    await Promise.resolve()

    expect(resume).not.toHaveBeenCalled()

    resolvePause?.()
    await resuming

    expect(resume).toHaveBeenCalledTimes(1)
  })

  it('does not resume a stale owner or resume twice', async () => {
    let resolveFirstPause: (() => void) | undefined

    const pause = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(
        () =>
          new Promise<void>(resolve => {
            resolveFirstPause = resolve
          })
      )
      .mockResolvedValue(undefined)

    const resume = vi.fn().mockResolvedValue(undefined)
    const coordinator = createWakePauseCoordinator({ pause, resume })

    coordinator.pause()
    await Promise.resolve()
    const firstResume = coordinator.resume()
    const duplicateResume = coordinator.resume()
    coordinator.pause()
    resolveFirstPause?.()
    const secondResume = coordinator.resume()
    await Promise.all([firstResume, duplicateResume, secondResume])

    expect(pause).toHaveBeenCalledTimes(2)
    expect(resume).toHaveBeenCalledTimes(1)
  })
})

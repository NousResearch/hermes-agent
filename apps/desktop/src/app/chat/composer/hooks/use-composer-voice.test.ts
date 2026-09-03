import { describe, expect, it, vi } from 'vitest'

import {
  acquireMicrophoneLease,
  disposeAssistantSubscriptions,
  runVoiceControllerCallback
} from './use-composer-voice'

describe('composer voice ownership', () => {
  it('keeps ownership and waits for pause settlement when aborted', async () => {
    let settlePause!: () => void
    let pauseSettled = false

    const pause = () =>
      new Promise<void>(resolve => {
        settlePause = () => {
          pauseSettled = true
          resolve()
        }
      })

    const resume = vi.fn(() => {
      expect(pauseSettled).toBe(true)
    })

    const controller = new AbortController()

    const acquired = acquireMicrophoneLease({
      owner: Symbol('owner'),
      pause,
      resume,
      signal: controller.signal,
      voiceContextIsCurrent: () => true
    })

    controller.abort()
    await Promise.resolve()
    expect(resume).not.toHaveBeenCalled()

    settlePause()
    await expect(acquired).resolves.toBeNull()
    expect(resume).toHaveBeenCalledOnce()
  })

  it('releases once and rearms only after the ownership barrier', async () => {
    let settlePause!: () => void
    let pauseSettled = false

    const pause = () =>
      new Promise<void>(resolve => {
        settlePause = () => {
          pauseSettled = true
          resolve()
        }
      })

    const resume = vi.fn(() => expect(pauseSettled).toBe(true))
    const acquired = acquireMicrophoneLease({ owner: Symbol('owner'), pause, resume, voiceContextIsCurrent: () => true })

    settlePause()
    const lease = await acquired
    lease?.release()
    lease?.release()

    expect(resume).toHaveBeenCalledOnce()
  })

  it('checks context before and after a delayed submit on session switch', async () => {
    let current = true
    let settleSubmit!: () => void
    const isCurrent = vi.fn(() => current)

    const submit = () =>
      new Promise<void>(resolve => {
        settleSubmit = resolve
      })

    const completion = runVoiceControllerCallback(isCurrent, submit)

    current = false
    settleSubmit()
    await completion

    expect(isCurrent).toHaveBeenCalledTimes(2)
  })

  it('proactively disposes assistant subscriptions on context replacement', () => {
    const first = vi.fn()
    const second = vi.fn()
    const disposers = new Set([first, second])

    disposeAssistantSubscriptions(disposers)

    expect(first).toHaveBeenCalledOnce()
    expect(second).toHaveBeenCalledOnce()
    expect(disposers.size).toBe(0)
  })
})

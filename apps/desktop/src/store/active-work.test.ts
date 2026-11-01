import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { HermesActiveWork } from '@/global'

import { createStreamThrottle } from '../../electron/stream-throttle'
import { $sessions } from './session'
import { clearAllSessionStates, publishSessionState } from './session-states'
import { $autoSpeakReplies } from './voice-prefs'
import { retainVoiceStreamPaint } from './voice-stream-paint'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const setActiveWork = vi.fn()

const busy = (storedSessionId: string, isBusy: boolean) =>
  ({ busy: isBusy, needsInput: false, storedSessionId }) as ClientSessionState

const session = (id: string, title: null | string) => ({ id, title }) as (typeof $sessions.value)[number]

beforeAll(async () => {
  desktopWindow.hermesDesktop = { setActiveWork } as unknown as Window['hermesDesktop']
  // Subscribes at import time, so the bridge has to exist first.
  await import('./active-work')
})

beforeEach(() => {
  setActiveWork.mockReset()
  $autoSpeakReplies.set(false)
  clearAllSessionStates()
  $sessions.set([])
  setActiveWork.mockClear()
})

describe('active work bridge', () => {
  it('reports a busy session by title', () => {
    $sessions.set([session('s1', 'Fix login'), session('s2', 'Idle chat')])
    publishSessionState('runtime-1', busy('s1', true))

    expect(setActiveWork).toHaveBeenLastCalledWith({ count: 1, titles: ['Fix login'] })
  })

  it('counts an untitled busy session without inventing a title', () => {
    $sessions.set([session('s1', null)])
    publishSessionState('runtime-1', busy('s1', true))

    expect(setActiveWork).toHaveBeenLastCalledWith({ count: 1, titles: [] })
  })

  it('drops back to nothing when the turn ends', () => {
    $sessions.set([session('s1', 'Fix login')])
    publishSessionState('runtime-1', busy('s1', true))
    publishSessionState('runtime-1', busy('s1', false))

    expect(setActiveWork).toHaveBeenLastCalledWith({ count: 0, titles: [] })
  })

  it('keeps only the minimized conversation owner painting until its last local lease is released', () => {
    const minimizedWindow = () => ({
      isDestroyed: () => false,
      isVisible: () => false,
      isMinimized: () => true,
      webContents: { isDestroyed: () => false, setBackgroundThrottling: vi.fn() }
    })
    const owner = minimizedWindow()
    const peer = minimizedWindow()
    const throttle = createStreamThrottle()
    throttle.register(owner)
    throttle.register(peer)
    // Exercise the real renderer summary and main-process policy, not a mocked summary.
    setActiveWork.mockImplementation((work: HermesActiveWork) => {
      throttle.setVoiceActive(owner, work.voiceActive === true)
      throttle.update(work.count > 0)
    })

    $autoSpeakReplies.set(true)
    $sessions.set([session('s1', ' Fix login '), session('s2', null)])
    publishSessionState('runtime-1', busy('s1', true))
    publishSessionState('runtime-2', busy('s2', true))
    const work = { count: 2, titles: ['Fix login'] }
    expect(setActiveWork).toHaveBeenLastCalledWith(work)
    expect(owner.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(true)
    expect(peer.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(true)

    const releaseFirst = retainVoiceStreamPaint()
    const releaseSecond = retainVoiceStreamPaint()
    try {
      try {
        expect(setActiveWork).toHaveBeenLastCalledWith({ ...work, voiceActive: true })
        expect(owner.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(false)
        expect(peer.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(true)
      } finally {
        releaseFirst()
      }

      expect(setActiveWork).toHaveBeenLastCalledWith({ ...work, voiceActive: true })
      expect(owner.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(false)
      expect(peer.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(true)
    } finally {
      releaseSecond()
    }

    expect(setActiveWork).toHaveBeenLastCalledWith(work)
    expect(owner.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(true)
    expect(peer.webContents.setBackgroundThrottling).toHaveBeenLastCalledWith(true)
    expect($autoSpeakReplies.get()).toBe(true)
  })
})

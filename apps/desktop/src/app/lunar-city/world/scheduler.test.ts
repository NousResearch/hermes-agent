import { describe, expect, it, vi } from 'vitest'

import { createNavigationController, type NavigationEntity } from './navigation'
import { createFrameScheduler, type FrameTick } from './scheduler'

function schedulerHarness() {
  const renderer = { stopRenderLoop: vi.fn() }
  const frames: FrameTick[] = []

  const onFrame = vi.fn((frame: FrameTick) => {
    frames.push(frame)

    return false
  })

  const requestFrame = vi.fn(() => 7)
  const cancelFrame = vi.fn()
  const setTimer = vi.fn(() => 8)
  const clearTimer = vi.fn()
  let now = 0

  const scheduler = createFrameScheduler({
    cancelFrame,
    captureMetrics: true,
    clearTimer,
    now: () => now,
    onFrame,
    renderer,
    requestFrame,
    setTimer
  })

  return {
    cancelFrame,
    clearTimer,
    frames,
    now: (value: number) => {
      now = value
    },
    onFrame,
    renderer,
    requestFrame,
    scheduler,
    setTimer
  }
}

describe('FrameScheduler', () => {
  it('caps interaction at 30 FPS for five seconds then returns to 15 FPS ambient cadence', () => {
    const harness = schedulerHarness()
    harness.scheduler.noteInteraction(0)

    for (let now = 0; now <= 5_000; now += 1) {
      harness.scheduler.tick(now)
    }

    const interactiveFrames = harness.frames.filter(frame => frame.targetFps === 30)
    expect(interactiveFrames.length).toBeLessThanOrEqual(151)
    expect(interactiveFrames.length).toBeGreaterThan(140)

    harness.scheduler.requestVisualAt(6_000)
    harness.scheduler.tick(6_000)

    expect(harness.frames.at(-1)).toMatchObject({ targetFps: 15 })
  })

  it('parks an unchanged city and draws ambient frames only when a visual deadline is due', () => {
    const harness = schedulerHarness()

    harness.scheduler.tick(0)
    harness.scheduler.requestVisualAt(100)
    harness.scheduler.tick(50)
    expect(harness.onFrame).not.toHaveBeenCalled()

    harness.scheduler.tick(100)
    expect(harness.onFrame).toHaveBeenCalledOnce()
    expect(harness.setTimer).toHaveBeenCalledOnce()
  })

  it('reaches zero frames while hidden and resumes with one current-snapshot frame', () => {
    const harness = schedulerHarness()
    harness.scheduler.noteInteraction(0)
    harness.scheduler.setVisible(false)
    harness.scheduler.tick(100)

    expect(harness.onFrame).not.toHaveBeenCalled()
    expect(harness.renderer.stopRenderLoop).toHaveBeenCalledOnce()
    expect(harness.cancelFrame).toHaveBeenCalledOnce()

    harness.scheduler.setVisible(true)
    harness.scheduler.tick(100)

    expect(harness.onFrame).toHaveBeenCalledOnce()
  })

  it('reports cumulative rendered timestamps and truthful zero target while paused', () => {
    const harness = schedulerHarness()
    harness.scheduler.requestRender()
    harness.scheduler.tick(100)
    harness.scheduler.requestRender()
    harness.scheduler.tick(200)

    expect(harness.scheduler.getMetrics()).toEqual({
      frameTimestampsMs: [100, 200],
      listeners: 0,
      rafs: 1,
      renderFrames: 2,
      targetFps: 15,
      timers: 0
    })

    harness.scheduler.setVisible(false)
    expect(harness.scheduler.getMetrics()).toEqual({
      frameTimestampsMs: [100, 200],
      listeners: 0,
      rafs: 0,
      renderFrames: 2,
      targetFps: 0,
      timers: 0
    })
  })

  it('cancels frame, timer, and later wake work on disposal', () => {
    const harness = schedulerHarness()
    harness.scheduler.noteInteraction(0)
    harness.scheduler.requestVisualAt(200)
    harness.scheduler.dispose()
    harness.scheduler.dispose()
    harness.scheduler.tick(500)

    expect(harness.cancelFrame).toHaveBeenCalledOnce()
    expect(harness.clearTimer).toHaveBeenCalledOnce()
    expect(harness.onFrame).not.toHaveBeenCalled()
  })

  it('never exceeds 30 FPS while dirty at a 120 Hz source cadence', () => {
    const harness = schedulerHarness()
    harness.scheduler.noteInteraction(0)

    for (let now = 0; now <= 1_000; now += 1000 / 120) {
      harness.scheduler.requestRender()
      harness.scheduler.tick(now)
    }

    expect(harness.frames.length).toBeLessThanOrEqual(30)
    expect(harness.frames.length).toBeGreaterThan(25)
  })

  it('parks repeated dirty requests behind one timer while waiting for the next allowed frame', () => {
    const harness = schedulerHarness()
    harness.scheduler.noteInteraction(0)
    harness.scheduler.tick(0)
    harness.requestFrame.mockClear()

    for (let now = 1; now < 20; now += 1) {
      harness.now(now)
      harness.scheduler.requestRender()
      harness.scheduler.tick(now)
    }

    expect(harness.requestFrame).not.toHaveBeenCalled()
    expect(harness.setTimer).toHaveBeenCalledWith(expect.any(Function), expect.any(Number))

    harness.scheduler.tick(34)

    expect(harness.frames.length).toBe(2)
  })

  it('keeps context loss paused when document visibility returns until every reason clears', () => {
    const harness = schedulerHarness()

    harness.scheduler.setPauseReason('document-hidden', true)
    harness.scheduler.setPauseReason('context-lost', true)
    harness.scheduler.setPauseReason('document-hidden', false)

    expect(harness.scheduler.targetFps(10)).toBe(0)
    expect(harness.renderer.stopRenderLoop).toHaveBeenCalledOnce()

    harness.scheduler.setPauseReason('context-lost', false)
    harness.scheduler.tick(10)

    expect(harness.scheduler.targetFps(10)).toBe(15)
    expect(harness.onFrame).toHaveBeenCalledOnce()
  })

  it('continues a worker over multiple frames without a camera transition and parks after its arrival', () => {
    const worker: NavigationEntity = {
      animation: 'idle',
      key: 'session:worker' as never,
      position: { x: 0, y: 0, z: 0 }
    }

    const navigation = createNavigationController({
      destinations: { review: { x: 1, y: 0, z: 0 } },
      query: {
        computePath: () => [
          { x: 0, y: 0, z: 0 },
          { x: 1, y: 0, z: 0 }
        ]
      },
      speedUnitsPerSecond: 1,
      workerClips: new Set(['idle', 'walk', 'work'])
    })

    const harness = schedulerHarness()
    harness.onFrame.mockImplementation(frame => {
      harness.frames.push(frame)

      return navigation.tick(frame.elapsedMs)
    })

    navigation.move(worker, 'review', 'work')
    harness.scheduler.requestRender()

    for (let now = 0; now <= 1_100; now += 34) {
      harness.scheduler.tick(now)
    }

    expect(worker.position).toEqual({ x: 1, y: 0, z: 0 })
    expect(worker.animation).toBe('work')
    expect(harness.frames.length).toBeGreaterThan(2)
    const completedFrames = harness.frames.length
    harness.scheduler.tick(2_000)
    expect(harness.frames).toHaveLength(completedFrames)
  })
})

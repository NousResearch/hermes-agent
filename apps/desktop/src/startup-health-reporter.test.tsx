import { act, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { scheduleAfterNextPaint, StartupHealthReporter } from './startup-health-reporter'

describe('StartupHealthReporter', () => {
  it('uses two animation frames so one paint completes before reporting', () => {
    const frames = new Map<number, FrameRequestCallback>()
    let nextFrame = 1
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      const id = nextFrame
      nextFrame += 1
      frames.set(id, callback)

      return id
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
    const reportHealthy = vi.fn()
    const cancel = scheduleAfterNextPaint(reportHealthy)

    expect([...frames.keys()]).toEqual([1])
    const firstFrame = frames.get(1)
    frames.delete(1)
    firstFrame?.(16)
    expect(reportHealthy).not.toHaveBeenCalled()
    expect([...frames.keys()]).toEqual([2])

    const secondFrame = frames.get(2)
    frames.delete(2)
    secondFrame?.(32)
    expect(reportHealthy).toHaveBeenCalledOnce()

    cancel()
    vi.unstubAllGlobals()
  })

  it('reports only after descendants mount and the first paint boundary is reached', async () => {
    const events: string[] = []
    let afterPaint: (() => void) | undefined

    const reportHealthy = vi.fn(() => {
      events.push('healthy')
    })

    function RequiredStartupDependency() {
      useEffect(() => {
        events.push('dependency-ready')
      }, [])

      return <div>ready</div>
    }

    render(
      <StartupHealthReporter
        reportHealthy={reportHealthy}
        scheduleAfterPaint={callback => {
          afterPaint = callback

          return () => {
            afterPaint = undefined
          }
        }}
      >
        <RequiredStartupDependency />
      </StartupHealthReporter>
    )

    expect(events).toEqual(['dependency-ready'])
    expect(reportHealthy).not.toHaveBeenCalled()

    await act(async () => afterPaint?.())

    expect(events).toEqual(['dependency-ready', 'healthy'])
    expect(reportHealthy).toHaveBeenCalledOnce()
  })

  it('cancels a pending report when the startup subtree unmounts', () => {
    const reportHealthy = vi.fn()
    let cancelled = false
    let afterPaint: (() => void) | undefined

    const view = render(
      <StartupHealthReporter
        reportHealthy={reportHealthy}
        scheduleAfterPaint={callback => {
          afterPaint = callback

          return () => {
            cancelled = true
          }
        }}
      >
        <div>ready</div>
      </StartupHealthReporter>
    )

    view.unmount()
    act(() => afterPaint?.())

    expect(cancelled).toBe(true)
    expect(reportHealthy).not.toHaveBeenCalled()
  })

  it('uses the root preload health bridge after paint', async () => {
    let afterPaint: (() => void) | undefined
    const signalStartupHealthy = vi.fn().mockResolvedValue({ ok: true })
    vi.stubGlobal('hermesDesktop', { signalStartupHealthy })

    const view = render(
      <StartupHealthReporter
        scheduleAfterPaint={callback => {
          afterPaint = callback

          return () => undefined
        }}
      >
        <div>ready</div>
      </StartupHealthReporter>
    )

    act(() => afterPaint?.())
    await waitFor(() => expect(signalStartupHealthy).toHaveBeenCalledOnce())

    view.unmount()
    vi.unstubAllGlobals()
  })
})

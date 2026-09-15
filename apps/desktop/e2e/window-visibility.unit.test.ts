import type { ElectronApplication, Page } from '@playwright/test'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { waitForPageWindowVisible } from './window-visibility'

function windows(targetVisible: boolean, auxiliaryVisible: boolean) {
  const target = { isVisible: vi.fn(() => targetVisible) }
  const auxiliary = { isVisible: vi.fn(() => auxiliaryVisible) }

  const handle = {
    evaluate: vi.fn(async (read: (window: typeof target) => boolean) => read(target)),
    dispose: vi.fn(async () => {})
  }

  const page = {
    waitForTimeout: vi.fn((delay: number) => new Promise(resolve => setTimeout(resolve, delay)))
  } as unknown as Page

  const app = {
    browserWindow: vi.fn(async (requested: Page) => {
      if (requested !== page) {
        throw new Error('Unexpected page requested')
      }

      return handle
    }),
    evaluate: vi.fn(async (read: (electron: { BrowserWindow: { getAllWindows: () => typeof target[] } }) => boolean) =>
      read({ BrowserWindow: { getAllWindows: () => [auxiliary, target] } }))
  }

  return { app, page, target, auxiliary, handle }
}

describe('waitForPageWindowVisible', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('accepts the visible page window even when an unrelated first window is hidden', async () => {
    const fixture = windows(true, false)
    const pending = waitForPageWindowVisible(fixture.app as unknown as ElectronApplication, fixture.page, 50)
    await vi.advanceTimersByTimeAsync(500)
    await pending
    expect(fixture.app.browserWindow).toHaveBeenCalledWith(fixture.page)
    expect(fixture.target.isVisible).toHaveBeenCalled()
    expect(fixture.auxiliary.isVisible).not.toHaveBeenCalled()
    expect(fixture.handle.dispose).toHaveBeenCalledOnce()
  })

  it('fails explicitly at the deadline when the page window stays hidden despite a visible auxiliary window', async () => {
    const fixture = windows(false, true)

    const outcome = waitForPageWindowVisible(fixture.app as unknown as ElectronApplication, fixture.page, 50)
      .then(() => 'resolved', error => String(error))

    await vi.advanceTimersByTimeAsync(500)
    expect(await outcome).toContain('Page BrowserWindow did not become visible within 50ms')
    expect(fixture.auxiliary.isVisible).not.toHaveBeenCalled()
    expect(fixture.handle.dispose).toHaveBeenCalledOnce()
  })

  it('does not turn a failed page-window lookup into a successful readiness result', async () => {
    const fixture = windows(true, true)
    fixture.app.browserWindow.mockRejectedValueOnce(new Error('Synthetic window lookup failed'))
    await expect(waitForPageWindowVisible(fixture.app as unknown as ElectronApplication, fixture.page, 50))
      .rejects.toThrow('Synthetic window lookup failed')
  })
})

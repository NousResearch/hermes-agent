import { expect, test, vi } from 'vitest'

import { registerDesktopActiveWorkRuntime } from './desktop-active-work-runtime'

test('renderer active-work reports share one quit snapshot and release throttling when a sender closes', () => {
  let report: ((event: any, payload: unknown) => void) | undefined
  let destroyed: (() => void) | undefined

  const ipcMain = { on: vi.fn((channel, handler) => {
    expect(channel).toBe('hermes:active-work')
    report = handler
  }) }

  const sender = { id: 9, once: vi.fn((event, handler) => {
    expect(event).toBe('destroyed')
    destroyed = handler
  }) }

  const runtime = registerDesktopActiveWorkRuntime(ipcMain as any)

  report?.({ sender }, { count: 1, titles: ['Current turn'] })
  expect(runtime.activeWorkByWebContents.get(9)).toEqual({ count: 1, titles: ['Current turn'] })
  expect(runtime.streamThrottle.isUnthrottled()).toBe(true)
  destroyed?.()
  expect(runtime.activeWorkByWebContents.size).toBe(0)
  expect(runtime.streamThrottle.isUnthrottled()).toBe(true) // trailing flush remains live
})

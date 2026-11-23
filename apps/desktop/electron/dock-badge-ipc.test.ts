import type { IpcMainInvokeEvent } from 'electron'
import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ app: { setBadgeCount: vi.fn(() => true) } }))

import { DOCK_BADGE_CHANNEL, dockBadgeCount, registerDockBadgeIpc } from './dock-badge-ipc'

describe('dock badge IPC', () => {
  it('paints the renderer count on the app icon and clears it with 0', () => {
    const handle = vi.fn()
    const setBadgeCount = vi.fn(() => true)
    registerDockBadgeIpc({ handle }, setBadgeCount)

    expect(handle).toHaveBeenCalledWith(DOCK_BADGE_CHANNEL, expect.any(Function))
    const set = handle.mock.calls[0][1] as (event: IpcMainInvokeEvent, value: unknown) => boolean

    set({} as IpcMainInvokeEvent, 3)
    expect(setBadgeCount).toHaveBeenLastCalledWith(3)

    set({} as IpcMainInvokeEvent, 0)
    expect(setBadgeCount).toHaveBeenLastCalledWith(0)
  })

  it('never passes a renderer value through unchecked', () => {
    expect(dockBadgeCount(2.7)).toBe(2)
    expect(dockBadgeCount(-1)).toBe(0)
    expect(dockBadgeCount('nope')).toBe(0)
    expect(dockBadgeCount(undefined)).toBe(0)
    expect(dockBadgeCount(Number.POSITIVE_INFINITY)).toBe(0)
    expect(dockBadgeCount(123456)).toBe(9999)
  })
})

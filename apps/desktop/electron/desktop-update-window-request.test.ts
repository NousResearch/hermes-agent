import { expect, test, vi } from 'vitest'

import { createDesktopUpdateWindowRequest } from './desktop-update-window-request'

test('update window request queues before renderer readiness and focuses the ready window', () => {
  let window: any = null
  const request = createDesktopUpdateWindowRequest({ getMainWindow: () => window })
  request.sendOpenUpdatesRequested()
  expect(request.getPendingOpenUpdates()).toBe(true)

  const send = vi.fn()
  const show = vi.fn()
  const focus = vi.fn()
  window = { isDestroyed: () => false, isVisible: () => false, show, focus,
    webContents: { isDestroyed: () => false, send } }
  request.setPendingOpenUpdates(false)
  request.setRendererReadyForDeepLink(true)
  request.sendOpenUpdatesRequested()

  expect(send).toHaveBeenCalledExactlyOnceWith('hermes:open-updates')
  expect(show).toHaveBeenCalledOnce()
  expect(focus).toHaveBeenCalledOnce()
  expect(request.getPendingOpenUpdates()).toBe(false)
})

import { afterEach, expect, it, vi } from 'vitest'

import { startNativeNotifyDismissal } from './native-notify-dismissal'
import { $activeSessionId } from './session'

afterEach(() => {
  $activeSessionId.set(null)
  vi.restoreAllMocks()
})

it("withdraws a session's OS notifications once the user is looking at it, never while away", () => {
  const dismissNotifications = vi.fn(async () => 0)
  Object.assign(window, { hermesDesktop: { ...window.hermesDesktop, dismissNotifications } })
  const focused = vi.spyOn(document, 'hasFocus').mockReturnValue(false)
  const stop = startNativeNotifyDismissal()

  // Switching sessions while the window is in the background is not seeing it.
  $activeSessionId.set('runtime-1')
  expect(dismissNotifications).not.toHaveBeenCalled()

  focused.mockReturnValue(true)
  window.dispatchEvent(new Event('focus'))
  expect(dismissNotifications).toHaveBeenCalledWith({ sessionId: 'runtime-1' })

  $activeSessionId.set('runtime-2')
  expect(dismissNotifications).toHaveBeenLastCalledWith({ sessionId: 'runtime-2' })

  stop()
  $activeSessionId.set('runtime-3')
  expect(dismissNotifications).toHaveBeenCalledTimes(2)
})

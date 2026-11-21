import { $activeSessionId } from './session'

// An OS notification is a request for attention. Once the user is looking at
// the session it is about (the window is focused and that chat is on screen),
// the request is answered: "Approval needed" for a prompt already approved
// in-app, or "Response ready" for a reply already read, must not linger in
// Notification/Action Center until the user swipes each one away.

function userIsLooking(): boolean {
  if (typeof document === 'undefined' || document.hidden) {
    return false
  }

  return typeof document.hasFocus !== 'function' || document.hasFocus()
}

/** Withdraw the active session's OS notifications when the user is looking at it. */
export function dismissSeenSessionNotifications(): void {
  const sessionId = $activeSessionId.get()

  if (sessionId && userIsLooking()) {
    void window.hermesDesktop?.dismissNotifications?.({ sessionId })?.catch(() => undefined)
  }
}

/** Track session switches and window returns; returns the teardown. */
export function startNativeNotifyDismissal(): () => void {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const unsubscribe = $activeSessionId.listen(() => dismissSeenSessionNotifications())
  window.addEventListener('focus', dismissSeenSessionNotifications)
  document.addEventListener('visibilitychange', dismissSeenSessionNotifications)

  return () => {
    unsubscribe()
    window.removeEventListener('focus', dismissSeenSessionNotifications)
    document.removeEventListener('visibilitychange', dismissSeenSessionNotifications)
  }
}

import { clearNotifications, notifyError } from '@/store/notifications'
import { setResumeExhaustedSessionId, setResumeFailedSessionId, setResumeReattachSessionId } from '@/store/session'

export interface ResumeAttemptOptions {
  authoritativeSnapshot?: boolean
  // use-resume-reattach's quiet retry of a chat that already shows its history.
  reattach?: boolean
}

/**
 * A fresh resume attempt (route change, retry, reconnect, reselect) retires
 * every latch an earlier attempt armed for the session: the self-heals in
 * use-route-resume / use-resume-reattach must stop treating it as stranded or
 * detached, and the chat view drops the exhausted error for the loader. This
 * attempt re-arms the one that matches how it ends.
 */
export function clearResumeLatches(storedSessionId: string): void {
  const retire = (current: null | string) => (current === storedSessionId ? null : current)

  setResumeFailedSessionId(retire)
  setResumeExhaustedSessionId(retire)
  setResumeReattachSessionId(retire)
}

export const resumeFailedToastId = (storedSessionId: string) => `resume-failed:${storedSessionId}`

/**
 * Opening a chat drops the toasts of the one before it. A re-attach retry
 * stays on the same chat and reruns up to four times; wiping every toast in
 * the app on each run would dismiss whatever else the user is reading.
 */
export function clearResumeToasts(options: ResumeAttemptOptions | undefined): void {
  if (!options?.reattach) {
    clearNotifications()
  }
}

/**
 * One "Resume failed" toast per incident, counted once. The first failure
 * shows it under a stable id; a re-attach retry is the same incident, so its
 * failure leaves that toast as it is instead of showing (and counting) it
 * again. use-resume-reattach dismisses it once a runtime binds.
 */
export function notifyResumeFailed(
  error: unknown,
  fallback: string,
  storedSessionId: string,
  options: ResumeAttemptOptions | undefined
): void {
  if (!options?.reattach) {
    notifyError(error, fallback, { id: resumeFailedToastId(storedSessionId) })
  }
}

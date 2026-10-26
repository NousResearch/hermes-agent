import { setResumeExhaustedSessionId, setResumeFailedSessionId, setResumeReattachSessionId } from '@/store/session'

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

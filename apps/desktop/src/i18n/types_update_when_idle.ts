/** Updates overlay: an in-app update while agent work is running. */
export interface UpdateWhenIdleCopy {
  busyTitle: string
  /** Lead-in above the list of work the update would stop. */
  busyBody: string
  chats: (count: number) => string
  /** A cron run in flight; `jobs` are the backend's job ids. */
  cron: (jobs: string) => string
  delegation: string
  input: string
  /** The backend could not prove it is idle (unreadable ledger or older runtime). */
  unknown: string
  /** Consent button: `count` is how many listed items the update interrupts. */
  updateNow: (count: number) => string
  updateWhenIdle: string
  waitingTitle: string
  waitingBody: string
  waitingFor: string
  /** Every reason cleared: the gated hand-off is about to run. */
  waitingIdle: string
  /** The wait's overall bound, as a local time. */
  waitingUntil: (time: string) => string
  cancel: string
  timedOut: string
}

/** Cron page: stopping a live run and the per-job missed-run (catch-up) policy. */
export interface CronControlCopy {
  /** Detail action that stops the job's run in flight. */
  stop: string
  /** The same action while the cancel is landing. */
  stopping: string
  /** Toast once the run booked its cancelled attempt. */
  stopped: string
  /** Toast when the run completed on its own before the cancel reached it. */
  finishedFirst: string
  /** Toast when no run was in flight by the time Stop was pressed. */
  nothingToStop: string
  failedStop: string
  /** Header note for a job whose newest attempt was stopped. */
  lastCancelled: (reason: string) => string
  /** Editor field: what to do with runs missed while Hermes was not running. */
  catchUpLabel: string
  catchUpHint: string
  catchUpChoices: { default: string; run: string; skip: string }
}

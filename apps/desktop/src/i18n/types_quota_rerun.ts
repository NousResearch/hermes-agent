/** A failed turn the backend armed to rerun itself at the provider's reset
 *  (tui_gateway/session_quota_resume.py; opt-in `desktop.quota_resume.enabled`). */
export interface QuotaRerunCopy {
  /** Shown on the error card instead of the manual "Retry when the limit resets". */
  scheduled: (time: string) => string
}

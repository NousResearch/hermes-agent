/** Inbox page: finished scheduled-job results, plus their new-result alerts. */
export interface InboxCopy {
  title: string
  /** Badge aria-label on the sidebar row. */
  unreadCount: (count: number) => string
  loading: string
  /** First list failed; shown above a Retry button. */
  loadFailed: string
  emptyTitle: string
  emptyDesc: string
  /** Chip on a run the scheduler is still executing. */
  running: string
  unread: string
  /** Stand-in when a run finished without response text. */
  noPreview: string
  refresh: string
  refreshing: string
  markReadFailed: string
  /** Alert title when several results landed at once. */
  newResultsTitle: (count: number) => string
  newResultsBody: string
}

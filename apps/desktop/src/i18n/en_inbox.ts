import type { InboxCopy } from './types_inbox'

export const enInbox: InboxCopy = {
  title: 'Inbox',
  unreadCount: (count: number) => `${count} unread`,
  loading: 'Loading inbox…',
  loadFailed: "Couldn't load the inbox",
  emptyTitle: 'No results yet',
  emptyDesc: 'When a scheduled job finishes, its result lands here.',
  running: 'Running',
  unread: 'Unread',
  noPreview: 'Finished without a response.',
  refresh: 'Refresh',
  refreshing: 'Refreshing…',
  markReadFailed: 'Could not mark the result as read',
  newResultsTitle: (count: number) => `${count} new scheduled job results`,
  newResultsBody: 'Open the Inbox to read them.'
}

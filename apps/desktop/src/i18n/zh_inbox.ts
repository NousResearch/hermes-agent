import type { InboxCopy } from './types_inbox'

export const zhInbox: InboxCopy = {
  title: '收件箱',
  unreadCount: (count: number) => `${count} 条未读`,
  loading: '正在加载收件箱…',
  loadFailed: '无法加载收件箱',
  emptyTitle: '暂无结果',
  emptyDesc: '定时任务完成后，结果会出现在这里。',
  running: '运行中',
  unread: '未读',
  noPreview: '已完成，但没有回复。',
  refresh: '刷新',
  refreshing: '正在刷新…',
  markReadFailed: '无法将结果标记为已读',
  newResultsTitle: (count: number) => `${count} 条新的定时任务结果`,
  newResultsBody: '打开收件箱查看。'
}

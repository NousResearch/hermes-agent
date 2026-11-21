import type { InboxCopy } from './types_inbox'

export const zhHantInbox: InboxCopy = {
  title: '收件匣',
  unreadCount: (count: number) => `${count} 則未讀`,
  loading: '正在載入收件匣…',
  loadFailed: '無法載入收件匣',
  emptyTitle: '尚無結果',
  emptyDesc: '排程工作完成後，結果會出現在這裡。',
  running: '執行中',
  unread: '未讀',
  noPreview: '已完成，但沒有回覆。',
  refresh: '重新整理',
  refreshing: '正在重新整理…',
  markReadFailed: '無法將結果標記為已讀',
  newResultsTitle: (count: number) => `${count} 則新的排程工作結果`,
  newResultsBody: '開啟收件匣查看。'
}

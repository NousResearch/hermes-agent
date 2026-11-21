import type { InboxCopy } from './types_inbox'

export const jaInbox: InboxCopy = {
  title: '受信トレイ',
  unreadCount: (count: number) => `未読 ${count} 件`,
  loading: '受信トレイを読み込み中…',
  emptyTitle: 'まだ結果はありません',
  emptyDesc: 'スケジュール済みジョブが完了すると、その結果がここに届きます。',
  running: '実行中',
  unread: '未読',
  noPreview: '応答なしで完了しました。',
  refresh: '更新',
  refreshing: '更新中…',
  markReadFailed: '結果を既読にできませんでした',
  newResultsTitle: (count: number) => `スケジュール済みジョブの新しい結果が ${count} 件あります`,
  newResultsBody: '受信トレイを開いて確認してください。'
}

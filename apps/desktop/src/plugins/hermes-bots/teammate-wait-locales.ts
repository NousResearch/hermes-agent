/**
 * Copy for the "Waiting on teammates" strip above a Bot Chat's composer
 * (`teammate-waits.tsx`). Its own bundle so the strip's keys sit together; wired
 * into `BotsMessages.waiting` for every locale Bot Mode ships.
 */

// A type alias, not an interface: the plugin bundle's index signature must accept it.
export type TeammateWaitMessages = {
  title: string
  queued: string
  claimed: string
  working: string
  waitingClarify: string
  waitingApproval: string
  failed: (reason: string) => string
  unknownOutcome: string
  open: string
  dismiss: string
  /** The do-not-resend rule, shown under a row whose outcome is unknown and as the strip's tooltip. */
  neverResent: string
}

const en: TeammateWaitMessages = {
  title: 'Waiting on teammates',
  queued: 'Queued',
  claimed: 'Picked up',
  working: 'Working on it',
  waitingClarify: 'Waiting on you — answer its question',
  waitingApproval: 'Waiting on you — approve or deny',
  failed: reason => `Failed: ${reason}`,
  unknownOutcome: 'Outcome unknown',
  open: 'Open',
  dismiss: 'Dismiss',
  neverResent:
    'Nothing here is ever resent. An unknown outcome means the teammate may already have acted — open its chat before asking again.'
}

const ja: TeammateWaitMessages = {
  title: 'チームメイトの応答待ち',
  queued: '待機中',
  claimed: '受付済み',
  working: '作業中',
  waitingClarify: 'あなたの回答待ち — 質問に答えてください',
  waitingApproval: 'あなたの承認待ち — 承認または拒否',
  failed: reason => `失敗: ${reason}`,
  unknownOutcome: '結果不明',
  open: '開く',
  dismiss: '閉じる',
  neverResent:
    'ここにある依頼は自動で再送されません。結果不明の場合、相手がすでに実行した可能性があります。もう一度頼む前にそのチャットを確認してください。'
}

const zh: TeammateWaitMessages = {
  title: '等待队友',
  queued: '排队中',
  claimed: '已接收',
  working: '处理中',
  waitingClarify: '等待你 — 请回答它的问题',
  waitingApproval: '等待你 — 请批准或拒绝',
  failed: reason => `失败：${reason}`,
  unknownOutcome: '结果未知',
  open: '打开',
  dismiss: '忽略',
  neverResent: '这里的请求永远不会自动重发。结果未知表示队友可能已经执行——再次请求前请先打开它的聊天查看。'
}

const zhHant: TeammateWaitMessages = {
  title: '等待隊友',
  queued: '排隊中',
  claimed: '已接收',
  working: '處理中',
  waitingClarify: '等待你 — 請回答它的問題',
  waitingApproval: '等待你 — 請核准或拒絕',
  failed: reason => `失敗：${reason}`,
  unknownOutcome: '結果未知',
  open: '開啟',
  dismiss: '忽略',
  neverResent: '這裡的請求永遠不會自動重送。結果未知表示隊友可能已經執行——再次請求前請先開啟它的聊天查看。'
}

export const TEAMMATE_WAIT_LOCALES = { en, ja, zh, 'zh-hant': zhHant }

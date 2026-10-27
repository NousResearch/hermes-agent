/** Copy for the Needs you band and the one-glyph row status (roster-status.ts).
 *  Kept beside the other topical bundles so i18n.ts stays a table of parts. */
export interface RosterStatusMessages {
  band: string
  needsYou: string
  asks: string
  wantsToRun: string
  blocked: string
  mentioned: string
  routine: string
  working: string
  failed: string
  unread: string
  idle: string
  stale: string
}

export const ROSTER_STATUS_LOCALES: Record<'en' | 'ja' | 'zh' | 'zh-hant', RosterStatusMessages> = {
  en: {
    band: 'Needs you',
    needsYou: 'Needs you',
    asks: 'Asks',
    wantsToRun: 'Wants to run',
    blocked: 'Delivery blocked — your call',
    mentioned: 'Mentioned you',
    routine: 'Routine running',
    working: 'Working',
    failed: 'Last turn failed',
    unread: 'New reply',
    idle: 'Idle',
    stale: 'Gateway unreachable — last known state'
  },
  ja: {
    band: '対応が必要',
    needsYou: '対応が必要',
    asks: '質問',
    wantsToRun: '実行の許可を求めています',
    blocked: '配信がブロックされました — 判断が必要です',
    mentioned: 'あなたをメンションしました',
    routine: 'ルーティン実行中',
    working: '作業中',
    failed: '前回のターンが失敗しました',
    unread: '新しい返信',
    idle: '待機中',
    stale: 'ゲートウェイに接続できません — 最後に確認した状態'
  },
  zh: {
    band: '需要你处理',
    needsYou: '需要你处理',
    asks: '提问',
    wantsToRun: '请求运行',
    blocked: '投递被拦截 — 需要你决定',
    mentioned: '提到了你',
    routine: '例行任务运行中',
    working: '工作中',
    failed: '上一轮失败',
    unread: '新回复',
    idle: '空闲',
    stale: '网关不可达 — 显示最后已知状态'
  },
  'zh-hant': {
    band: '需要你處理',
    needsYou: '需要你處理',
    asks: '提問',
    wantsToRun: '請求執行',
    blocked: '投遞被攔截 — 需要你決定',
    mentioned: '提到了你',
    routine: '例行任務執行中',
    working: '工作中',
    failed: '上一輪失敗',
    unread: '新回覆',
    idle: '閒置',
    stale: '閘道無法連線 — 顯示最後已知狀態'
  }
}

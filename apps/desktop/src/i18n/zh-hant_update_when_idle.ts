import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const zhHantUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes 仍在工作',
  busyBody: '更新會重新啟動 Hermes,並停止:',
  chats: (n: number) => `${n} 個對話正在執行`,
  cron: (jobs: string) => `排程工作正在執行(${jobs})`,
  delegation: '背景子代理正在執行',
  input: '有提示正在等待你的回答',
  unknown: 'Hermes 無法確認目前閒置',
  updateNow: (n: number) => `立即更新(中斷 ${n} 項)`,
  updateWhenIdle: '閒置時更新',
  waitingTitle: '更新將等待 Hermes 完成工作',
  waitingBody: '沒有工作執行時,更新會自動開始。在此之前可繼續正常使用 Hermes。',
  waitingFor: '正在等待:',
  waitingIdle: '目前沒有工作執行,正在開始更新…',
  waitingUntil: (time: string) => `如果到 ${time} Hermes 仍在忙,將略過本次更新。`,
  cancel: '取消更新',
  timedOut: 'Hermes 一直在忙,因此未執行更新。沒有任何變更,請稍後再試。'
}

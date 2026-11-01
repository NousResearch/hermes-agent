import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const zhUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes 仍在工作',
  busyBody: '更新会重启 Hermes,并停止:',
  chats: (n: number) => `${n} 个对话正在运行`,
  cron: (jobs: string) => `定时任务正在运行(${jobs})`,
  delegation: '后台子代理正在运行',
  input: '有提示正在等待你的回答',
  unknown: 'Hermes 无法确认当前空闲',
  updateNow: (n: number) => `立即更新(中断 ${n} 项)`,
  updateWhenIdle: '空闲时更新',
  waitingTitle: '更新将等待 Hermes 完成工作',
  waitingBody: '没有任务运行时,更新会自动开始。在此之前可继续正常使用 Hermes。',
  waitingFor: '正在等待:',
  waitingIdle: '当前没有任务运行,正在开始更新…',
  waitingUntil: (time: string) => `如果到 ${time} Hermes 仍在忙,将跳过本次更新。`,
  cancel: '取消更新',
  timedOut: 'Hermes 一直在忙,因此未执行更新。没有任何更改,请稍后再试。'
}

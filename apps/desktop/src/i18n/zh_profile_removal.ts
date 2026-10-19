// The profile delete / reset confirmation copy, composed by zh.ts as `profiles.removal`.
import type { TranslationOverrides } from './define-locale'

export const zhProfileRemoval: NonNullable<TranslationOverrides['profiles']>['removal'] = {
  deleteTitle: '删除配置档案？',
  deleteDescPrefix: '这将删除 ',
  deleteDescMid: ' 并移除其 ',
  deleteDescSuffix: ' 目录。此操作无法撤销。',
  deleting: '删除中…',
  resetTitle: '重置默认配置档案？',
  resetDesc:
    '默认配置档案无法删除：它的目录包含 Hermes 安装和你的其他配置档案。' +
    '重置会清除它的记忆、技能、设置、SOUL.md 和聊天，并移除它的定时任务。' +
    '你的其他配置档案、Hermes 安装以及 API 密钥和登录信息都会保留。此操作无法撤销。',
  resetAction: '重置',
  resetting: '正在重置...',
  resetDone: '配置档案已重置'
}

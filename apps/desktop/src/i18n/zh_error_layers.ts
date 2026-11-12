import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's layer fallbacks (lib/error-surface-copy.ts): used when a failure code has no own entry.
// Partial like the rest of this locale: an untranslated layer falls back to English.
export const zhErrorLayers: {
  [K in 'errorLayerBodies' | 'errorLayers']: Partial<ThreadCopy[K]>
} = {
  errorLayers: {
    auth: '认证错误',
    billing: '额度不足',
    disk: '磁盘已满',
    endpoint: '自定义端点错误',
    gateway: '无法发送你的消息',
    generic: '本轮失败',
    provider: '模型服务商错误',
    runtime: '本地运行时错误',
    streaming: '流式连接错误'
  },
  errorLayerBodies: {
    generic: 'Hermes 回复时出现问题。请重试；若问题持续，请复制错误详情。',
    provider: 'AI 服务无法完成此请求。请稍后重试或切换服务商。',
    endpoint: 'Hermes 无法连接到你的自定义模型服务器。请确认它正在运行，然后重新发送消息。',
    streaming: '回复完成前连接已断开。请重试以重新发送。'
  }
}

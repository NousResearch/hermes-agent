import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const zhMessagingGateway = {
  hintGatewayStopped: '在此启动消息网关以建立连接。',
  restartNeeded: '已保存。请重启消息网关以应用新设置。',
  restartNow: '立即重启',
  restarting: '正在重启…',
  restartFailedManual: '网关重启失败 — 请手动重启并检查网关日志。',
  restartFailedManualDetail: '请再次尝试重启；如果仍然失败，请打开日志并发送诊断信息。',
  startMessagingGateway: '启动消息网关',
  startingMessagingGateway: '正在启动消息网关…',
  gatewayStartFailed: '消息网关启动失败。'
} satisfies MessagingGatewayTranslations

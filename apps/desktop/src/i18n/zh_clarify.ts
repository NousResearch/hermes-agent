import type { ClarifyCopy } from './types_clarify'

export const zhClarify: Partial<ClarifyCopy> = {
  notReady: '澄清请求尚未就绪',
  gatewayDisconnected: 'Hermes 网关未连接',
  sendFailed: '无法发送澄清响应',
  loadingQuestion: '正在加载问题…',
  other: '其他 (输入你的答案)',
  placeholder: '输入你的答案…',
  skip: '跳过',
  skipped: '已跳过',
  noAnswer: '未回答',
  confirmAndContinueLabel: '确认并继续',
  singleSelectHint: '选一个',
  multiSelectHint: '可多选',
  oneQuestion: '1 个问题',
  questionProgress: (answered, total) => `已回答 ${answered}/${total}`,
  notDelivered: '此问题未送达应用，无法在此回答。请按停止结束本轮，然后在聊天中回复。'
}

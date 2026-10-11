import type { Translations } from './types'

export const zhModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: '此模型没有可用选项',
  options: '选项',
  thinking: '思考',
  fast: '快速',
  ultrafast: 'Ultrafast',
  useStandardSpeed: '使用标准速度',
  auto: '自动',
  cold: '冷启动',
  speedPolicy: '速度策略',
  effort: '推理强度',
  minimal: '最小',
  low: '低',
  medium: '中',
  high: '高',
  xhigh: '极高',
  max: '最高',
  ultra: '超高',
  sendsOnRoute: (level: string) => `此路由实际发送 ${level}`,
  updateFailed: '模型选项更新失败',
  fastFailed: '快速模式更新失败'
}

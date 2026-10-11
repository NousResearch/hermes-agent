import type { Translations } from './types'

export const jaModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: 'このモデルにはオプションがありません',
  options: 'オプション',
  thinking: '思考',
  fast: '高速',
  ultrafast: 'Ultrafast',
  useStandardSpeed: '標準速度を使用',
  auto: '自動',
  cold: 'コールド',
  speedPolicy: '速度ポリシー',
  effort: '努力度',
  minimal: '最小',
  low: '低',
  medium: '中',
  high: '高',
  xhigh: '特高',
  max: '最大',
  ultra: 'ウルトラ',
  sendsOnRoute: (level: string) => `このルートでは ${level} を送信`,
  updateFailed: 'モデルオプションの更新に失敗しました',
  fastFailed: '高速モードの更新に失敗しました'
}

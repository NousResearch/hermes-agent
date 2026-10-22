// Per-turn status-bar readout copy (缓存命中率 / 吞吐 / 会话费用).
// Split out of zh.ts, which the code-health line cap may only reduce.
import type { StatusbarReadoutTranslations } from './types_statusbar_readouts'

export const zhStatusbarReadouts: StatusbarReadoutTranslations = {
  cacheHitRateTitle: '本会话的提示缓存命中率 — 缓存 token 更便宜，越高越省',
  sessionCostTitle: '本会话到目前为止的花费。~ 前缀表示按供应商价格估算，最终账单可能不同；$0.00 表示订阅已包含这些调用',
  tokensPerSecondTitle: '每秒输出 token 数，取最近 10 次模型调用的平均值',
  toggleCacheHitRate: '缓存命中率',
  toggleSessionCost: '会话费用',
  toggleTokensPerSecond: '每秒 token 数'
}

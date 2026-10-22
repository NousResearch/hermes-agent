// Per-turn status-bar readout copy: cache hit rate, throughput, session spend.
// Split out of en.ts, which the code-health line cap may only reduce.
import type { StatusbarReadoutTranslations } from './types_statusbar_readouts'

export const enStatusbarReadouts: StatusbarReadoutTranslations = {
  cacheHitRateTitle: 'Prompt cache hit rate this session — cached tokens cost less, so higher is cheaper',
  sessionCostTitle:
    'Spend on this session so far. A ~ prefix is our estimate from provider pricing — the final charge can differ; $0.00 means no charge was recorded for these calls',
  tokensPerSecondTitle: 'Output tokens per second, averaged over the last 10 model calls',
  toggleCacheHitRate: 'Cache hit rate',
  toggleSessionCost: 'Session cost',
  toggleTokensPerSecond: 'Tokens per second'
}

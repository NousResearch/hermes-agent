import type { QuotaRerunCopy } from './types_quota_rerun'

export const jaQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `制限がリセットされる ${time} に Hermes がこれを再実行します`
}

import type { QuotaRerunCopy } from './types_quota_rerun'

export const zhHantQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes 將在 ${time} 額度重設時重新執行此任務`
}

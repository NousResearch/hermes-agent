import type { QuotaRerunCopy } from './types_quota_rerun'

export const enQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes will rerun this at ${time}, when the limit resets`
}

import type { QuotaRerunCopy } from './types_quota_rerun'

export const deQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes führt dies um ${time} erneut aus, wenn das Limit zurückgesetzt wird`
}

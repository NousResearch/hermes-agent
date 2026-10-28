import type { QuotaRerunCopy } from './types_quota_rerun'

export const ruQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes повторит это в ${time}, когда лимит сбросится`
}

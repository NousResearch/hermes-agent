import type { QuotaRerunCopy } from './types_quota_rerun'

export const arQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `سيعيد Hermes تشغيل هذا عند ${time}، عند إعادة تعيين الحد`
}

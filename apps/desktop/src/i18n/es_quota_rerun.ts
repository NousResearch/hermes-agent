import type { QuotaRerunCopy } from './types_quota_rerun'

export const esQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes volverá a ejecutar esto a las ${time}, cuando se restablezca el límite`
}

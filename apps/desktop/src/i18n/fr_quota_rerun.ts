import type { QuotaRerunCopy } from './types_quota_rerun'

export const frQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes relancera ceci à ${time}, lorsque la limite sera réinitialisée`
}

import type { QuotaRerunCopy } from './types_quota_rerun'

export const zhQuotaRerun: QuotaRerunCopy = {
  scheduled: time => `Hermes 将在 ${time} 限额重置时重新运行此任务`
}

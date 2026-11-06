/** Provider account usage copy shared by the statusbar and its detail panel. */
export interface AccountUsagePanelCopy {
  apiKeyUsage: string
  bankedResets: (count: number) => string
  creditsBalance: string
  creditsUnlimited: string
  extraUsage: string
  extraUsageValue: (used: string, limit: string) => string
  modelQuota: string
  ofLimitRemaining: (remaining: string, limit: string) => string
  plan: (plan: string) => string
  refresh: string
  refreshFailed: string
  remaining: (remaining: string) => string
  resetIntervals: {
    daily: string
    monthly: string
    weekly: string
  }
  resets: (time: string) => string
  sessionLine: (input: string, output: string, total: string) => string
  thisSession: string
  title: string
  unavailable: string
  updated: (time: string) => string
  updatedUnknown: string
  usageThisMonth: (value: string) => string
  usageThisWeek: (value: string) => string
  usageToday: (value: string) => string
  usageTotal: (value: string) => string
  used: (used: string) => string
  windowLabels: {
    api_key_quota: string
    current_session: string
    current_week: string
    opus_week: string
    session: string
    sonnet_week: string
    weekly: string
  }
}

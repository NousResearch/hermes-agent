// Settings > About > Storage; `Translations.settings.storageSection`.
export interface StorageSectionTranslations {
  title: string
  description: (path: string) => string
  total: string
  refresh: string
  scanning: string
  failed: string
  empty: string
  truncated: string
  folder: string
  file: string
  openLocation: string
  openHint: string
  back: string
  home: string
  cleanupTitle: string
  cleanupSessions: string
  cleanupAge: string
  cleanupDays: (days: number) => string
  cleanupCounting: string
  cleanupPreviewFailed: string
  cleanupNothing: (days: number) => string
  cleanupMatched: (count: number, days: number) => string
  cleanupAction: string
  cleanupConfirmTitle: (count: number) => string
  cleanupConfirmBody: (days: number) => string
  cleanupConfirmAction: (count: number) => string
  cleanupProfile: string
  cleanupLastUsed: string
  cleanupDone: (count: number) => string
  cleanupFailed: string
  cleanupFootnote: string
}

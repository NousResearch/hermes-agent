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
}

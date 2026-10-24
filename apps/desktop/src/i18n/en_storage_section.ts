import type { StorageSectionTranslations } from './types_storage_section'

// Settings > About > Storage, composed by en.ts as `settings.storageSection`.
export const enStorageSection: StorageSectionTranslations = {
  title: 'Storage',
  description: path => `Disk space used by Hermes in ${path}. This page only shows sizes; it never deletes anything.`,
  total: 'Total',
  refresh: 'Rescan',
  scanning: 'Measuring disk usage…',
  failed: 'Could not measure disk usage.',
  empty: 'Nothing found.',
  truncated: 'The scan stopped early on a very large folder, so these sizes are lower bounds.',
  folder: 'Folder',
  file: 'File',
  openLocation: 'Open file location',
  openHint: 'Click a folder to look inside; click its icon to show where it is.',
  back: 'Back',
  home: 'Hermes home'
}

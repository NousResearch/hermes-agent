import type { StorageSectionTranslations } from './types_storage_section'

// Settings > About > Storage, composed by en.ts as `settings.storageSection`.
export const enStorageSection: StorageSectionTranslations = {
  title: 'Storage',
  description: path => `Disk space used by Hermes in ${path}.`,
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
  home: 'Hermes home',
  cleanupTitle: 'Free up space',
  cleanupSessions: 'Inactive sessions',
  cleanupAge: 'Inactive for',
  cleanupDays: days => `Unused for ${days} days`,
  cleanupCounting: 'Counting…',
  cleanupPreviewFailed: 'Could not count inactive sessions.',
  cleanupNothing: days => `No ended sessions have been unused for ${days} days.`,
  cleanupMatched: (count, days) =>
    `${count} ended session${count === 1 ? '' : 's'} unused for ${days} days. Pinned and running sessions are kept.`,
  cleanupAction: 'Delete',
  cleanupConfirmTitle: count => `Permanently delete ${count} inactive session${count === 1 ? '' : 's'}?`,
  cleanupConfirmBody: days =>
    `Ended sessions with no activity in the last ${days} days are removed with their transcripts. This cannot be undone.`,
  cleanupConfirmAction: count => `Delete ${count}`,
  cleanupProfile: 'Profile',
  cleanupLastUsed: 'Last used',
  cleanupDone: count => `Deleted ${count} session${count === 1 ? '' : 's'}.`,
  cleanupFailed: 'Could not delete inactive sessions',
  cleanupFootnote:
    'Freed history is reused by new sessions inside state.db. To shrink the file itself, run `hermes sessions optimize` with Hermes closed.'
}

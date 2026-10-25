/** Desktop Settings -> Sessions copy (archived list, auto-archive, A2A recents).
 *
 * Split out of `types.ts`, whose line budget may only go down.
 */
export interface SettingsSessionsCopy {
  loading: string
  archivedTitle: string
  archivedIntro: string
  emptyArchivedTitle: string
  emptyArchivedDesc: string
  unarchive: string
  deletePermanently: string
  messages: (count: number) => string
  restored: string
  deleteConfirm: (title: string) => string
  autoArchiveTitle: string
  autoArchiveDesc: string
  autoArchiveDaysLabel: string
  autoArchiveDaysUnit: string
  autoArchiveFailed: string
  recentsShowA2ATitle: string
  recentsShowA2ADesc: string
  recentsShowA2AFailed: string
  defaultDirTitle: string
  defaultDirDesc: string
  defaultDirUpdated: string
  defaultsTo: (label: string) => string
  change: string
  choose: string
  clear: string
  notSet: string
  failedLoad: string
  unarchiveFailed: string
  deleteFailed: string
  updateDirFailed: string
  clearDirFailed: string
}

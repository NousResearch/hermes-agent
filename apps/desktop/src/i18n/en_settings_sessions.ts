import type { SettingsSessionsCopy } from './types_settings_sessions'

/** Desktop Settings -> Sessions copy. */
export const enSettingsSessions: SettingsSessionsCopy = {
  loading: 'Loading archived sessions…',
  archivedTitle: 'Archived sessions',
  archivedIntro:
    'Archived chats are hidden from the sidebar but keep all their messages. Alt/⌥+Shift-click a chat in the sidebar to archive it.',
  emptyArchivedTitle: 'Nothing archived',
  emptyArchivedDesc: 'Archive a chat to hide it here.',
  unarchive: 'Unarchive',
  deletePermanently: 'Delete permanently',
  messages: count => `${count} ${count === 1 ? 'message' : 'messages'}`,
  restored: 'Restored',
  deleteConfirm: title => `Permanently delete "${title}"? This cannot be undone.`,
  autoArchiveTitle: 'Auto-archive stale chats',
  autoArchiveDesc:
    "Automatically archive chats you haven't touched in a while. Pinned chats are never archived, and nothing is deleted — archived chats just move here.",
  autoArchiveDaysLabel: 'Archive after',
  autoArchiveDaysUnit: 'days of inactivity',
  autoArchiveFailed: 'Could not update auto-archive',
  recentsShowA2ATitle: 'Show A2A sessions in the recent list',
  recentsShowA2ADesc:
    'Tasks another machine dispatched to this one over A2A. They stay out of the sidebar by default because every inbound task adds a row that is not one of your conversations. Session search still finds them either way.',
  recentsShowA2AFailed: 'Could not update the A2A recents setting',
  defaultDirTitle: 'Default project directory',
  defaultDirDesc:
    'New sessions start in this folder unless you pick another. Leave it unset to use your home directory.',
  defaultDirUpdated: 'Default project directory updated — start a new chat (Ctrl/⌘+N) for it to take effect',
  defaultsTo: label => `Defaults to ${label}.`,
  change: 'Change',
  choose: 'Choose',
  clear: 'Clear',
  notSet: 'Not set',
  failedLoad: 'Could not load archived sessions',
  unarchiveFailed: 'Unarchive failed',
  deleteFailed: 'Delete failed',
  updateDirFailed: 'Could not update default directory',
  clearDirFailed: 'Could not clear default directory'
}

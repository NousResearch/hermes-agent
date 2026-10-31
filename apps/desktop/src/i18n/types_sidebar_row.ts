/** Sidebar session actions, activity labels and age copy. */
export interface SidebarRowTranslations {
  pin: string
  unpin: string
  markUnread: string
  markRead: string
  unreadFailed: string
  copyId: string
  export: string
  branchFrom: string
  rename: string
  archive: string
  unarchive: string
  newWindow: string
  openInTerminal: string
  hideTabBar: string
  openInNewTab: string
  openInSplit: string
  copyIdFailed: string
  sessionActions: string
  sessionRunning: string
  needsInput: string
  waitingForAnswer: string
  finishedUnread: string
  backgroundRunning: string
  activity: string
  draftSession: string
  handoffOrigin: (platform: string) => string
  continuationOrigin: string
  ownedByProfile: (profile: string) => string
  renamed: string
  renameFailed: string
  renameTitle: string
  renameDesc: string
  untitledPlaceholder: string
  deleteTitle: string
  deleteDesc: (title: string) => string
  deleting: string
  deleted: string
  untitledChat: (id: string) => string
  messageCount: (count: number) => string
  todoProgress: string
  ageNow: string
  ageDay: string
  ageHour: string
  ageMin: string
}

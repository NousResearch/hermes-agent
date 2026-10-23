// The right sidebar's copy (file tree, folder picker, terminals); `Translations.rightSidebar`.
export interface RightSidebarTranslations {
  aria: string
  panelsAria: string
  files: string
  terminal: string
  noFolderSelected: string
  changeCwdTitle: string
  remotePickerTitle: string
  remotePickerDescription: string
  remotePickerSelect: string
  remotePickerNewFolder: string
  remotePickerFolderName: string
  remotePickerCreateFolder: string
  remotePickerInvalidFolderName: string
  remotePickerCreateFolderFailed: (error: string) => string
  folderTip: (cwd: string) => string
  openFolder: string
  refreshTree: string
  collapseAll: string
  showIgnored: string
  hideIgnored: string
  previewUnavailable: string
  couldNotPreview: (path: string) => string
  noProjectTitle: string
  noProjectBody: string
  noProjectOpen: string
  noDiffs: string
  unreadableTitle: string
  unreadableBody: (error: string) => string
  emptyTitle: string
  emptyBody: string
  treeErrorTitle: string
  treeErrorBody: string
  tryAgain: string
  loadingTree: string
  loadingFiles: string
  terminalHide: string
  terminalsAria: string
  terminalNew: string
  terminalReadOnly: string
  terminalReadOnlyHelp: string
  terminalOpenInteractive: string
  terminalCloseOthers: string
  terminalCloseAll: string
  terminalDropUploadFailed: (name: string) => string
  terminalDropTooLargeTitle: string
  terminalDropTooLarge: (names: string, limit: string) => string
  terminalDropControlTitle: string
  terminalDropControl: (paths: string) => string
  addToChat: string
}

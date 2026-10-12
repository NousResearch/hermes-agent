// The sidebar projects section's copy shape; `Translations.sidebar.projects`.
export interface SidebarProjectsTranslations {
  showAllSessions: string
  sectionLabel: string
  home: string
  autoDiscovered: string
  newButton: string
  createTitle: string
  createDesc: string
  renameTitle: string
  addFolderTitle: string
  namePlaceholder: string
  foldersLabel: string
  ideaLabel: string
  ideaPlaceholder: string
  ideaGenerate: string
  ideaGenerating: string
  ideaShuffle: string
  noFolders: string
  addFolder: string
  primaryBadge: string
  removeFolder: string
  create: string
  menu: string
  menuRename: string
  menuAppearance: string
  noColor: string
  menuAddFolder: string
  menuSetActive: string
  menuDelete: string
  moveToProject: string
  movedTo: (name: string) => string
  moveFailed: string
  moveNoFolder: string
  moveNoProjects: string
  reveal: string
  copyPath: string
  removeFromSidebar: string
  createdInPreviousContext: string
  hiddenFromSidebar: string
  undoHide: string
  createFailed: string
  staleBackend: string
  deleteConfirm: string
  startWork: string
  newWorktreeTitle: string
  newWorktreeDesc: string
  branchPlaceholder: string
  branchOff: () => { after: string; before: string }
  baseBranchPlaceholder: string
  baseBranchNone: string
  startWorkFailed: string
  worktreeStaleBackend: string
  worktreeProjectLabel: string
  worktreeProjectPlaceholder: string
  worktreeProjectNone: string
  convertBranch: string
  convertBranchTitle: string
  convertBranchDesc: string
  convertBranchPlaceholder: string
  convertBranchInstead: string
  branchOpenExisting: string
  branchSwitchHome: string
  branchCreateWorktree: string
  branchTrackRemote: string
  branchesLoading: string
  noBranches: string
  removeWorktree: string
  removeWorktreeFailed: string
  removeWorktreeConfirm: string
  removeWorktreeDirty: string
  forceRemove: string
  enter: (label: string) => string
  reorder: (label: string) => string
  toggle: (label: string, open: boolean) => string
  showAllCount: (count: number) => string
  back: string
}

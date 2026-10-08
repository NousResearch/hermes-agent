export interface CodingTranslations {
  diffUnified: string
  diffSplit: string
  diffBefore: string
  diffAfter: string
  title: string
  noBranch: string
  detached: string
  clean: string
  changed: (count: number) => string
  ahead: (count: number) => string
  behind: (count: number) => string
  review: string
  close: string
  openChanges: string
  openFile: string
  stage: string
  unstage: string
  stageAll: string
  viewAsTree: string
  viewAsList: string
  revert: string
  revertAll: string
  revertConfirm: string
  revertAllConfirm: string
  staged: string
  noChanges: string
  notRepo: string
  noDiff: string
  scopeUncommitted: string
  scopeBranch: string
  scopeLastTurn: string
  readOnlyScope: string
  commit: string
  commitAndPush: string
  commitPlaceholder: (shortcut: string) => string
  generateCommitMessage: string
  stopGenerating: string
  createPr: string
  openPr: string
  ghMissing: string
  agentShip: string
  agentShipUnavailable: string
  agentShipPrompt: string
  newBranch: string
  branchOffFrom: (base: string) => string
  switchTo: (branch: string) => string
  switchFailed: (branch: string) => string
  worktrees: string
}

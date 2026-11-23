// Settings ▸ Backup & Transfer; `Translations.settings.backup`.
export interface BackupTransferTranslations {
  intro: string
  /** ⌘K rows that open this page. */
  backupCommand: string
  transferCommand: string
  backupTitle: string
  backupHelp: string
  backup: string
  backupRunning: string
  backupSaved: string
  backupKept: (path: string) => string
  backupFailed: string
  transferTitle: string
  transferHelp: string
  signInTitle: string
  signInHelp: string
  signIn: string
  /** The Webapp has no portal window: sign-in happens in Settings ▸ Providers. */
  signInWebappHelp: string
  openProviders: string
  pickOrg: string
  chooseAgent: string
  provisioning: string
  down: string
  noAgentsTitle: string
  noAgentsHelp: string
  openPortal: string
  newAgentTitle: string
  newAgentHelp: string
  confirmTitle: (agent: string) => string
  replacesTitle: string
  replaces: string[]
  keepsTitle: string
  keeps: string[]
  botNote: (agent: string) => string
  undoNote: (agent: string) => string
  acknowledge: (agent: string) => string
  replace: (agent: string) => string
  cancel: string
  stepBackup: string
  stepUpload: string
  stepRestore: (agent: string) => string
  doneTitle: string
  doneHelp: (agent: string) => string
  /** `bots`: the platforms the transfer switched off, already joined for the locale ("Telegram and Discord"). */
  botsOff: (bots: string, agent: string) => string
  /** The restore finished but the cloud agent's gateway did not start again. */
  gatewayDown: (agent: string) => string
  undo: string
  undoing: string
  openAgent: string
  undoneTitle: string
  undoneHelp: (agent: string) => string
  startAgain: string
  errorTitle: string
  partialHelp: (agent: string) => string
  retry: string
}

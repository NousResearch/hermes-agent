// The shape of the command center's maintenance section copy.
export interface MaintenanceTranslations {
  runOps: string
  doctor: string
  doctorDesc: string
  securityAudit: string
  securityAuditDesc: string
  backup: string
  backupDesc: string
  debugShare: string
  debugShareDesc: string
  debugShareRunning: string
  debugShareLinks: string
  debugShareFailed: string
  copyLink: string
  linkCopied: string
  curator: string
  curatorDesc: string
  curatorDescWithBuiltins: string
  curatorPaused: string
  curatorActive: string
  curatorDisabled: string
  curatorLastRun: (when: string) => string
  curatorNeverRan: string
  pause: string
  resume: string
  runNow: string
  memoryData: string
  memoryDataDesc: string
  memoryProvider: (name: string) => string
  builtinMemory: string
  memoryFile: string
  userFile: string
  bytes: (size: string) => string
  empty: string
  resetMemory: string
  resetUser: string
  resetAll: string
  resetConfirm: (target: string) => string
  resetDone: (files: string) => string
  resetFailed: string
  actionStarted: (name: string) => string
  actionFailed: (name: string) => string
  running: string
  viewLog: string
}

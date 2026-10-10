export interface CommandCenterTranslations {
  /** Usage → "This month": month-to-date usage per provider against its budget. */
  thisMonth: string
  monthProgress: (day: number, days: number) => string
  tokensUsed: (tokens: string) => string
  spentKnown: (amount: string) => string
  unpricedSessions: (count: number) => string
  budgetOf: (used: string, limit: string) => string
  projected: (percent: number) => string
  runsOutOn: (date: string) => string
  setBudget: string
  budgetTokens: string
  budgetUsd: string
  saveBudget: string
  savingBudget: (seconds: number) => string
  clearBudget: string
  budgetSaveFailed: (error: string) => string
  loadingMonth: (seconds: number) => string
  noUsageThisMonth: string
  /** Money left on the provider account, beside its month row. */
  balanceLeft: (amounts: string, age: string) => string
  balanceUnknown: string
  balanceUnavailable: string
  close: string
  paletteTitle: string
  back: string
  searchPlaceholder: string
  goTo: string
  goToSession: string
  branches: string
  projects: string
  openFolder: string
  openFolderAt: (path: string) => string
  newSessionInProject: (project: string) => string
  commands: string
  startInBranch: (branch: string) => string
  commandCenter: string
  appearance: string
  settings: string
  changeTheme: string
  changeColorMode: string
  pets: {
    title: string
    placeholder: string
    loading: string
    error: string
    staleBackend: string
    empty: string
    turnOff: string
    turnOn: string
    installed: string
    generatedTag: string
    adoptFailed: string
    toggleFailed: (enabled: boolean) => string
    noneAvailable: string
  }
  generatePet: {
    title: string
    placeholder: string
    promptHint: string
    readyHint: string
    generate: string
    generating: string
    retry: string
    hatch: string
    spawning: string
    hatching: string
    hatchingSub: string
    hatched: string
    hatchRow: (state: string, done: number, total: number) => string
    hatchComposing: string
    hatchSaving: string
    namePlaceholder: string
    staleBackend: string
    backgroundHint: string
    slowProviderHint: string
    remix: string
    remixConfirmTitle: string
    remixConfirmBody: string
    genericError: string
    referenceImageTooLarge: string
    referenceImageInvalid: string
    adopt: string
    startOver: string
  }
  installTheme: {
    title: string
    pageTitle: string
    placeholder: string
    loading: string
    error: string
    empty: string
    install: string
    installing: string
    installed: string
    installs: (count: string) => string
  }
  settingsFields: string
  mcpServers: string
  archivedChats: string
  sections: Record<'maintenance' | 'sessions' | 'system' | 'usage', string>
  nav: Record<'newChat' | 'settings' | 'capabilities' | 'messaging' | 'artifacts', { title: string; detail: string }>
  sectionEntries: Record<'sessions' | 'system' | 'usage', { title: string; detail: string }>
  providerNavigate: string
  providerSessions: string
  refresh: string
  refreshing: string
  noResults: string
  pinSession: string
  unpinSession: string
  exportSession: string
  deleteSession: string
  noSessions: string
  gatewayRunning: string
  gatewayStopped: string
  hermesActiveSessions: (version: string, count: number) => string
  restartGateway: string
  openBrowser: string
  toggleBrowser: string
  gatewayRestartFailed: string
  sharedGatewayRestartTitle: string
  sharedGatewayRestartDescription: (bots: string) => string
  sharedGatewayRestartConfirm: string
  sharedGatewayRestarted: (count: number) => string
  updateHermes: string
  reloadWindow: string
  actionRunning: string
  actionDone: string
  actionFailed: string
  actionStartedWaiting: string
  loadingStatus: string
  recentLogs: string
  noLogs: string
  days: (count: number) => string
  statSessions: string
  statApiCalls: string
  statTokens: string
  statCost: string
  actualCost: (cost: string) => string
  loadingUsage: string
  noUsage: (period: number) => string
  retry: string
  dailyTokens: string
  input: string
  output: string
  noDailyActivity: string
  topModels: string
  noModelUsage: string
  topSkills: string
  noSkillActivity: string
  actions: (count: string) => string
  logFile: string
  logLevel: string
  logSearchPlaceholder: string
  maintenance: {
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
}

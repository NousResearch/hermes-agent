export interface ConnectorsTranslations {
  connectors: {
    title: string
    connect: string
    skip: string
    cancel: string
    retry: string
    grant: string
    connected: string
    checking: string
    waitingSignIn: string
    notConnected: string
    notAvailable: string
    startWith: (count: number) => string
    startWithout: string
    skipped: string
    disabled: string
    failed: string
    needsAuth: string
    opening: string
    waiting: string
    timeout: string
    refresh: string
    statusError: string
    connectError: string
    connectErrorFor: (app: string) => string
    unavailable: string
    ownerMissing: string
    search: string
    empty: string
    disclaimer: string
    execution: string
  }
  sessionImport: {
    title: string
    subtitle: string
    action: string
    readingFrom: string
    connectedComputer: string
    destination: string
    all: string
    search: string
    scanning: string
    scanError: string
    scanHelp: string
    empty: string
    emptyHelp: string
    noMatches: string
    searchHelp: string
    skipped: string
    more: string
    messages: string
    choose: string
    chooseHelp: string
    previewLoading: string
    previewError: string
    previewHelp: string
    previewLimit: string
    you: string
    snapshot: string
    copyNotice: string
    importing: string
    open: string
    continue: string
    importError: string
  }
}

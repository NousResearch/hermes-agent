// The boot screen's string surface (including the update-hold screen); `Translations.boot`.
export interface BootTranslations {
  ready: string
  desktopBootFailedWithMessage: (message: string) => string
  steps: {
    connectingGateway: string
    loadingSettings: string
    loadingSessions: string
    retryingRemoteBackend: string
    startingDesktopConnection: string
    startingHermesDesktop: string
  }
  errors: {
    backgroundExited: string
    backgroundExitedDuringStartup: string
    backendStopped: string
    restartHermes: string
    openLogs: string
    desktopBootFailed: string
    gatewayConnectionLost: string
    gatewayConnectionLostDetail: string
    reconnectNow: string
    connectionSettings: string
    gatewaySignInRequired: string
    gatewaySignInRequiredDetail: string
    signInAgain: string
    ipcBridgeUnavailable: string
  }
  causes: {
    exitedEarly: string
    timedOut: string
    permission: string
    diskFull: string
    portInUse: string
    installMissing: string
  }
  failure: {
    title: string
    description: string
    details: string
    remoteTitle: string
    remoteDescription: string
    retry: string
    repairInstall: string
    useLocalGateway: string
    gatewaySettings: string
    back: string
    openLogs: string
    repairHint: string
    bundledReinstallHint: string
    reinstallApp: string
    remoteSignInHint: (signInLabel: string) => string
    signOutAndSignIn: string
    remoteFailureHint: string
    cloudDownTitle: string
    cloudDownDescription: string
    cloudDownHint: string
    cloudDownCheckPortal: string
    cloudDownDiscord: string
    hideRecentLogs: string
    showRecentLogs: string
    signedInTitle: string
    signedInMessage: string
    signInIncompleteTitle: string
    signInIncompleteMessage: string
    signInFailed: string
    signInToRemoteGateway: string
    signInWithProvider: (provider: string) => string
    identityProvider: string
  }
  // The blocked boot screen while an earlier update still holds the install (R8 D3).
  updateHold: {
    title: string
    titleUnverified: string
    description: string
    heldByProcess: (pid: number) => string
    heldUnknown: string
    unverified: string
    since: (time: string) => string
    lastChecked: (time: string) => string
    recoveryHint: string
    checkAgain: string
    quit: string
    openLogs: string
    startAnyway: string
    confirmTitle: string
    confirmBody: string
    confirmKeepWaiting: string
    confirmStart: string
    startAnywayRefused: string
  }
}

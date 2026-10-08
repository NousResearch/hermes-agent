// The updates surface (About page, updates overlay, statusbar version items); `Translations.updates`.
export interface UpdatesTranslations {
  discontinuedTitle: string
  discontinuedBody: string
  channels: { stable: string; canary: string }
  bundleSwapPending: string
  bundleSwapPendingDesc: string
  bundleSwapPendingAction: string
  stages: Record<string, string>
  checking: string
  checkFailedTitle: string
  tryAgain: string
  notAvailableTitle: string
  unsupportedMessage: string
  connectionRetry: string
  gitUnusable: string
  connectionSettings: string
  openDownloadPage: string
  latestBody: string
  latestBodyBackend: string
  allSetTitle: string
  availableTitle: string
  availableBody: string
  availableTitleBackend: string
  availableBodyBackend: string
  availableBodyNoChangelog: string
  availableBodyAppInstaller: string
  updateNow: string
  maybeLater: string
  moreChanges: (count: number) => string
  copyFullLog: string
  manualTitle: string
  manualUnavailableTitle: string
  manualBody: string
  manualBodyBackend: string
  manualPickedUp: string
  manualPickedUpBackend: string
  /** GUI/backend skew (#45205): backend updated but the running desktop app
   *  package (AppImage/.deb/.rpm) was not changed and must be reinstalled. */
  guiSkewTitle: string
  guiSkewBody: string
  copy: string
  copied: string
  done: string
  applyingBody: string
  applyingBodyBackend: string
  applyingClose: string
  applyingBodyAppInstaller: string
  applyingCloseAppInstaller: string
  checkUnknownTitleAppInstaller: string
  checkUnknownBodyAppInstaller: string
  errorTitle: string
  errorBody: string
  blockerTitle: string
  blockerBody: string
  foreignBlockerTitle: string
  foreignBlockerBody: string
  mixedBlockerBody: string
  closePreviewsAndUpdate: string
  closePreviewsAndCheckAgain: string
  localPreview: string
  portLabel: (port: number) => string
  pidLabel: (pid: number) => string
  technicalDetails: string
  notNow: string
  /** Multi-target update flow: client nudge after a backend update, and
   *  per-row fan-out outcomes when updating every registered instance. */
  clientAlsoBehindTitle: string
  clientAlsoBehindMessage: string
  clientAlsoBehindAction: string
  everythingDispatched: string
  everythingSkipped: string
  everythingRowFailed: string
  everythingFanoutFailedTitle: string
  changeLogNew: string
  changeLogFixed: string
  changeLogFaster: string
  changeLogImproved: string
  changeLogOther: string
  changeLogFallbackLabel: string
  changeLogFallbackItem: string
  applyStatus: {
    preparing: string
    pulling: string
    restarting: string
    notAvailable: string
    failed: string
    noReturn: string
    owed: (steps: string) => string
  }
  /** Update-status overlay + version-details (mechanism-aware update UI), read off t.updates directly. */
  appName: string
  version: (value: string) => string
  versionUnavailable: string
  checkNow: string
  seeWhatsNew: string
  releaseNotes: string
  onLatest: string
  installing: string
  cantReach: string
  tapCheck: string
  updateReady: (count: number) => string
  updateReadyUnknown: string
  localBranchBehind: (count: number) => string
  localBranchBehindUnknown: string
  localBranchCurrent: string
  availableBodyRelease: (tag: string) => string
  lastChecked: (age: string) => string
  never: string
  justNow: string
  minAgo: (count: number) => string
  hoursAgo: (count: number) => string
  daysAgo: (count: number) => string
  justNowSuffix: string
  bundleOutOfSync: string
  bundleOutOfSyncDesc: string
  bundleOutOfSyncAction: string
  checkingShort: string
  releaseAvailable: (tag: string) => string
  versionDetailsTitle: string
  versionDetailsBody: string
  versionDetailsVersion: string
  versionDetailsCommit: string
  versionDetailsBuildOrigin: string
  versionDetailsDistribution: string
  versionDetailsDistributionDesktop: string
  versionDetailsDistributionDesktopMsix: string
  versionDetailsDistributionDesktopInstaller: string
  versionDetailsDistributionSourceInstaller: string
  versionDetailsDistributionSourceInstallerDesktop: string
  versionDetailsDistributionSource: string
  versionDetailsDistributionSourceDesktop: string
  versionDetailsDistributionStore: string
  versionDetailsRuntime: string
  versionDetailsRuntimeEmbedded: string
  versionDetailsRuntimeExternal: string
  versionDetailsInstallId: string
  versionDetailsUncommittedChanges: string
}

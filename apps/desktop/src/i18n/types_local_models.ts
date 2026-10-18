export interface LocalModelsTranslations {
  connectionChanged: string
  title: string
  runtimeTitle: string
  runtimeReady: (backend: string) => string
  serverRunning: string
  runtimeInstalled: string
  runtimeInstalledDetail: (tag: string, backend: string) => string
  installTitle: string
  installDetail: string
  installAction: string
  installing: string
  installFailed: string
  hardwareTitle: string
  hardwareLoading: string
  vram: (label: string) => string
  ram: (label: string) => string
  unifiedMemory: string
  modelsTitle: string
  recommended: string
  /** Recommended-badge tooltip by resolver branch; unknown keys (newer
   *  backend) simply show no tooltip. */
  recommendedReason: Record<string, string>
  noRecommendationTitle: string
  noRecommendationDetail: string
  noRecommendationAction: string
  downloaded: string
  downloadAction: (size: string) => string
  downloadProgress: (done: string, total: string) => string
  downloadStatusRunning: string
  downloadSpeed: (rate: string) => string
  downloadEta: (time: string) => string
  /** Duration units the ETA is composed from; hours carries its
   *  remainder so a locale orders the two parts itself. */
  downloadEtaSeconds: (count: number) => string
  downloadEtaMinutes: (count: number) => string
  downloadEtaHours: (hours: number, minutes: number) => string
  downloadPausedLabel: string
  downloadPauseAction: string
  downloadResumeAction: string
  downloadDiscardAction: string
  downloadRetryingLabel: string
  downloadWaitingLabel: string
  downloadVerifyingLabel: string
  downloadInterruptedLabel: string
  downloadInterruptedEmptyLabel: string
  setupInterruptedLabel: string
  downloadDiscardDetail: string
  downloadDiscardFailed: (model: string) => string
  downloadDoneToast: (model: string) => string
  installDoneToast: string
  quickstartTitle: string
  quickstartDetail: (model: string, size: string) => string
  quickstartDetailReady: (model: string) => string
  quickstartAction: string
  quickstartConfigure: string
  quickstartDoneToast: (model: string) => string
  quickstartFailed: string
  quickstartStageEngine: string
  quickstartStageModel: string
  quickstartStageFinish: string
  useAction: string
  activePill: string
  updateTitle: string
  updateDetail: (next: string, current: string) => string
  updateAction: string
  updating: string
  upToDateTitle: string
  upToDateDetail: (tag: string, backend: string) => string
  activeDetail: string
  activeNotLoaded: string
  loadedPill: string
  placementResident: string
  placementSpilled: string
  placementResidentTip: string
  placementSpilledTip: string
  loadingPill: string
  ejectTip: string
  ejected: string
  ejectFailed: string
  stopServer: string
  startServer: string
  runtimeRunningDetail: string
  serverStopped: string
  serverStarted: string
  serverStopFailed: string
  serverStartFailed: string
  activating: string
  activateFailed: (model: string) => string
  activateDoneToast: (model: string) => string
  downloadFailed: (model: string) => string
  downloadPauseFailed: (model: string) => string
  downloadResumeFailed: (model: string) => string
  pillFitsGpu: string
  pillUsesRam: string
  pillTooBig: string
  browseTitle: string
  browseHint: string
  browsePlaceholder: string
  browseSearching: string
  browseListing: string
  browseShowFiles: string
  browseRefresh: string
  browseDownloads: string
  browseLikes: string
  browseGated: string
  browseNoGguf: string
  browseFitUnknown: string
  browseAlreadyDownloaded: string
  addedByYou: string
  browseDownloadStarted: string
  browseDownloadAria: string
  sideloadButton: string
  sideloadTitle: string
  sideloadDone: string
  sideloadAlreadyPresent: string
  pillFullContext: (max: string) => string
  pillFullContextTip: string
  pillUpTo: (max: string) => string
  pillGrowsTip: string
  pillVision: string
  deleteAction: string
  deleteConfirm: (model: string) => string
  deleted: (model: string) => string
  deleteFailed: string
}

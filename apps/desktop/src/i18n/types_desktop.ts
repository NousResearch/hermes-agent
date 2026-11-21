// The desktop surface's string copy; `Translations.desktop`.
export interface DesktopTranslations {
  audioReadFailed: string
  sessionUnavailable: string
  createSessionFailed: string
  sendFailedTitle: string
  sendFailedBody: string
  promptFailed: string
  providerCredentialRequired: string
  emptySlashCommand: string
  slashCommandIgnoredTitle: string
  slashCommandIgnoredBody: string
  desktopCommands: string
  skillCommandsAvailable: (count: number) => string
  warningLine: (message: string) => string
  yoloArmed: string
  yoloOff: string
  yoloSystem: (active: boolean) => string
  yoloTitle: string
  yoloToggleFailed: string
  profileStatus: (current: string) => string
  unknownProfile: string
  noProfileNamed: (target: string, available: string) => string
  newChatsProfile: (name: string) => string
  setProfileFailed: string
  sttDisabled: string
  stopFailed: string
  regenerateFailed: string
  editFailed: string
  editTurnUnavailable: string
  resumeFailed: string
  readOnlyTranscriptTitle: string
  readOnlyTranscriptBody: string
  readOnlyTranscriptSendBlocked: string
  resumeStrandedTitle: string
  resumeStrandedBody: string
  poolSlotTimeoutBody: string
  poolSlotTimeoutOpenSettings: string
  resumeRetry: string
  nothingToBranch: string
  branchNeedsChat: string
  sessionBusy: string
  branchStopCurrent: string
  branchNoText: string
  branchTitle: (n: number) => string
  branchFailed: string
  deleteFailed: string
  archived: string
  archiveFailed: string
  restored: string
  unarchiveFailed: string
  cwdChangeFailed: string
  cwdStagedTitle: string
  cwdStagedMessage: string
  modelSwitchConfirmBody: string
  modelSwitchConfirmLabel: string
  modelSwitchConfirmTitle: (model: string) => string
  modelSwitchConfirmTitleFallback: string
  modelSwitchFailed: string
  modelSwitchKeepLabel: string
  modelSwitchStaleNotice: string
  hydrationSyncing: (profile: string) => string
  sessionExported: string
  sessionExportFailed: string
  imageSaved: string
  downloadStarted: string
  restartToUseSaveImage: string
  restartToSaveImages: string
  imageDownloadFailed: string
  openImage: string
  downloadImage: string
  savingImage: string
  zoomIn: string
  zoomOut: string
  resetZoom: string
  imagePreviewFailed: string
  imageAttach: string
  imageWriteFailed: string
  imageAttachFailed: string
  pastedContent: string
  pasteAttachFailed: string
  attachImages: string
  clipboard: string
  noClipboardImage: string
  clipboardPasteFailed: string
  dropFiles: string
  handoff: {
    pickPlatform: string
    success: (platform: string) => string
    systemNote: (platform: string) => string
    failed: (error: string) => string
    timedOut: string
    startMessaging: string
  }
}

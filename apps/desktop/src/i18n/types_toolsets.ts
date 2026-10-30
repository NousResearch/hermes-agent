export interface ToolsetTranslations {
  loadingConfig: string
  savedTitle: string
  savedMessage: (key: string) => string
  removedTitle: string
  removedMessage: (key: string) => string
  failedSave: (key: string) => string
  failedRemove: (key: string) => string
  failedReveal: (key: string) => string
  removeConfirm: (key: string) => string
  set: string
  notSet: string
  selectedTitle: string
  selectedMessage: (provider: string) => string
  failedSelect: (provider: string) => string
  failedLoad: string
  noProviderOptions: string
  noProviders: string
  ready: string
  needsSignIn: string
  needsSetup: string
  activeBackend: string
  activeBackendHint: string
  browserAxesHint: string
  useBackend: string
  nousIncluded: string
  nousAuthNeededTitle: string
  nousAuthNeededMessage: (provider: string) => string
  nousAuthSignIn: string
  nousAuthDoneTitle: string
  nousAuthDoneMessage: string
  nousAuthFailed: string
  nousAuthFailedMessage: string
  nousAuthTryAgain: string
  noApiKeyRequired: string
  postSetupHint: (step: string) => string
  postSetupInstalledHint: string
  postSetupRun: string
  postSetupRerun: string
  postSetupInstalled: string
  postSetupRunning: string
  postSetupStarting: string
  postSetupCompleteTitle: string
  postSetupCompleteMessage: (step: string) => string
  postSetupErrorTitle: string
  postSetupErrorMessage: (step: string) => string
  postSetupOpenLogs: string
  postSetupRunAgain: string
  postSetupFailed: (step: string) => string
  webSearchActive: (backend: string) => string
  webExtractActive: (backend: string) => string
  webCapabilityUnset: string
  webUseForSearch: string
  webUseForExtract: string
  webUsedForSearch: string
  webUsedForExtract: string
  webCapabilitySelectedMessage: (provider: string, capability: string) => string
  failedSelectCapability: (provider: string) => string
  loadingModels: string
  modelSectionTitle: string
  modelCount: (count: number) => string
  modelInUse: string
  modelDefault: string
  modelInactiveHint: string
  modelSelectedTitle: string
  modelSelectedMessage: (model: string) => string
  failedSelectModel: (model: string) => string
  terminalBackend: {
    sectionTitle: string
    loading: string
    failedLoad: string
    ready: string
    needsSetup: string
    unavailable: string
    inUse: string
    selectedTitle: string
    selectedMessage: (backend: string) => string
    failedSelect: (backend: string) => string
    needsSetupHint: string
    needsSetupConfirmTitle: (backend: string) => string
    needsSetupConfirmDescription: (detail: string) => string
    needsSetupConfirmDescriptionGeneric: string
    needsSetupConfirmAction: string
    unavailableTitle: string
    unavailableMessage: (backend: string) => string
    openBackendSettings: string
    useLocal: string
    switchedToLocal: string
  }
  browserRealProfile: {
    label: string
    description: string
    enabledTitle: string
    enabledMessage: string
    disabledTitle: string
    disabledMessage: string
    failedSave: string
    prompt: {
      title: string
      body: string
      bulletSnapshot: string
      bulletLiveProfile: string
      bulletLocal: string
      dontShowAgain: string
      notNow: string
      enable: string
    }
  }
}

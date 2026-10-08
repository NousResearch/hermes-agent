/** Settings › Memory provider discovery, install and connect copy. */
export interface MemoryDiscoveryTranslations {
  installed: string
  availableToInstall: string
  installationRequired: string
  reviewInstall: string
  exploreAll: string
  missing: string
  installConsent: string
  builtin: string
  configureElsewhere: string
  notReady: string
  useFailed: string

  activeProvider: (name: string) => string
  useProvider: string
  loadFailed: string
  ownerChanged: string
  notDiscovered: string
  installedNotice: string
  backToMemory: string
  connect: string
  reconnect: string
  connectOAuth: string
  apiKeySet: string
  oauthSet: string
  waitingConsent: string
  stopWaiting: string
  stoppedWaiting: string
  startFailed: string
  connectionFailed: string
}

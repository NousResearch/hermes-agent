export interface OnboardingTranslations {
  headerTitle: string
  headerDesc: string
  preparingInstall: string
  starting: string
  lookingUpProviders: string
  collapse: string
  otherProviders: string
  haveApiKey: string
  chooseLater: string
  recommended: string
  connected: string
  featuredPitch: string
  fireworksPitch: string
  localModelsTitle: string
  localModelsPitch: string
  openRouterPitch: string
  apiKeyOptions: Record<string, { short: string; description: string }>
  backToSignIn: string
  getKey: string
  replaceCurrent: string
  pasteApiKey: string
  localApiKeyPlaceholder: string
  localModelNamePlaceholder: string
  couldNotSave: string
  connecting: string
  update: string
  flowSubtitles: Record<string, string>
  startingSignIn: (provider: string) => string
  verifyingCode: (provider: string) => string
  connectedProvider: (provider: string) => string
  connectedPicking: (provider: string) => string
  signInFailed: string
  signInExpired: string
  signInDidNotFinish: (provider: string) => string
  tryAgain: string
  useApiKeyInstead: string
  errorDetails: string
  pickDifferentProvider: string
  signInWith: (provider: string) => string
  openedBrowser: (provider: string) => string
  authorizeThere: string
  copyAuthCode: string
  pasteAuthCode: string
  reopenAuthPage: string
  waitingAuthorize: string
  externalPending: (provider: string) => string
  signedIn: string
  deviceCodeOpened: (provider: string) => string
  reopenVerification: string
  copy: string
  defaultModel: string
  freeTier: string
  pro: string
  free: string
  price: (input: string, output: string) => string
  change: string
  startChatting: string
  docs: (provider: string) => string
}

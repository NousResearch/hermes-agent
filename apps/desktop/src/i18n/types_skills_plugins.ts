// The Skills ▸ Plugins tab's copy shape; `Translations.skills.plugins`.
export interface SkillsPluginsTranslations {
  agentTitle: string
  agentBlurb: string
  pageBlurb: string
  halfDesktop: string
  halfDesktopHint: string
  halfAgent: string
  halfAgentIn: (profile: string) => string
  defaultProfile: string
  kindAgent: string
  kindDesktop: string
  kindBoth: string
  installAgentHere: string
  installAgentHereTip: (profile: string) => string
  installAgentHereNoOrigin: string
  desktopHalfPending: string
  desktopHalfPendingTip: string
  desktopHalfRemote: string
  desktopHalfRemoteTip: string
  emptyAll: string
  empty: string
  emptyHint: string
  loadFailed: string
  toggleFailed: (name: string) => string
  toolsetOn: (name: string, profile: string) => string
  toolsetOff: (name: string, profile: string) => string
  toolsetToggleFailed: (name: string) => string
  legacyBackend: string
  portableBadge: string
  serverStates: {
    connected: string
    app_not_running: string
    hermes_not_connected: string
    endpoint_unavailable: string
    no_interactive_session: string
    version_too_old: string
    missing_app: string
    unsupported_gpu: string
    unknown: string
  }
  catalogTitle: string
  catalogBrowse: string
  catalogHide: string
  catalogHint: string
  alreadyInstalled: (name: string) => string
  catalogProvenance: (sha: string) => string
  pinnedProvenance: (sha: string) => string
  pinnedBadge: (sha: string) => string
  update: string
  updateToPin: (sha: string) => string
  updateFailed: (name: string) => string
  updated: (name: string) => string
  upToDate: (name: string) => string
  updateConsentTitle: (name: string) => string
  updateConsentBody: (name: string) => string
  updateConsentConfirm: string
  uninstall: string
  uninstallTip: (name: string, profile: string) => string
  uninstallConfirmTitle: (name: string) => string
  uninstallConfirmBody: (name: string, profile: string) => string
  uninstallFailed: (name: string) => string
  uninstalled: (name: string) => string
  uninstallDesktopTip: (name: string) => string
  uninstallDesktopConfirmBody: (name: string) => string
  uninstalledDesktop: (name: string) => string
  deepLinkErrorTitle: string
  deepLinkCatalogInvalidName: string
  deepLinkCatalogUnknown: (name: string) => string
  deepLinkCatalogUnavailable: string
  settingsToggle: (name: string) => string
  settingsForm: {
    save: string
    saved: (name: string) => string
    saveFailed: (name: string) => string
    required: string
    secretSet: string
    secretStoredAs: (env: string) => string
  }
}

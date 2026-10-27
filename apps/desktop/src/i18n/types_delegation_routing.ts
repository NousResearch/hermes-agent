export interface DelegationRoutingTranslations {
  title: string
  settingsTitle: string
  intro: string
  modes: Record<'default' | 'model_map' | 'profile_routes', { label: string; description: string }>
  inUse: string
  plugin: string
  pluginMissing: string
  pluginDisabled: string
  pluginNotLoaded: string
  installPlugin: string
  installUnavailable: string
  enablePlugin: string
  enableFailed: string
  modeFailed: string
  loadFailed: string
  /** The saved mode cannot run, so core refuses every subagent ("Subagents will not start until …"). */
  blockedPill: string
  blockedDetail: string
  blockedMissing: string
  blockedDisabled: string
  blockedNotLoaded: string
  blockedUnavailable: (label: string) => string
  blockedUnregistered: (name: string) => string
  /** Picker row of another plugin's backend that ships no description of its own. */
  thirdPartyDescription: string

  modelRoutesTitle: string
  modelRoutesEmpty: string
  addRoute: string
  editRoute: string
  makeDefault: string
  deleteRoute: string
  defaultBadge: string
  routeActions: (name: string) => string
  deleteConfirmTitle: (name: string) => string
  deleteConfirmBody: string
  deleteConfirmNextDefault: (name: string) => string
  deleteConfirmAction: string
  saveFailed: string
  baseUrlRoute: (url: string) => string

  dialogNew: string
  dialogEdit: string
  dialogDesc: string
  nameLabel: string
  namePlaceholder: string
  nameHint: string
  descriptionLabel: string
  descriptionPlaceholder: string
  descriptionHint: string
  providerLabel: string
  providerPlaceholder: string
  modelLabel: string
  effortLabel: string
  effortInherit: string
  tierLabel: string
  tierNormal: string
  tierFast: string
  tierUltrafast: string
  tierHint: string
  optional: string
  errors: { required: string; spaces: string; duplicate: string }
  saveRoute: string
  createRoute: string

  profileRoutesTitle: string
  profileRoutesHint: string
  profileDescriptionPlaceholder: string
  profileDefault: string
  profileEmpty: string
  profileMissing: (name: string) => string

  inactiveTitle: string
  inactiveModelMap: (route: string) => string
  inactiveProfiles: (profile: string) => string
  inactiveModelMapNoDefault: string
  inactiveProfilesNoDefault: string
  inactivePlugin: (label: string) => string
  useDefault: string
}

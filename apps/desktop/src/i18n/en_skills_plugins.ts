import type { SkillsPluginsTranslations } from './types_skills_plugins'

/** English copy for the Skills ▸ Plugins tab; `skills.plugins`, composed by en.ts. */
export const enSkillsPlugins: SkillsPluginsTranslations = {
  agentTitle: 'Agent plugins',
  agentBlurb:
    'Extend the agent for the selected profile — tools, hooks, providers. Take effect after a gateway restart.',
  pageBlurb: 'A plugin can extend this app, the agent, or both — each half has its own switch.',
  halfDesktop: 'Desktop',
  halfDesktopHint: 'this app, same for every profile',
  halfAgent: 'Agent',
  halfAgentIn: (profile: string) => `Agent in ${profile}`,
  defaultProfile: 'Hermes (default)',
  kindAgent: 'Agent',
  kindDesktop: 'Desktop',
  kindBoth: 'Agent + Desktop',
  installAgentHere: 'Install here',
  installAgentHereTip: (profile: string) =>
    `The desktop half is loaded in this app, but the agent half is not installed in ${profile}. Install it there.`,
  installAgentHereNoOrigin:
    'The agent half is not installed in this profile, and this package was copied in by hand (no catalog entry or git remote), so it cannot be installed from here. Copy its folder into the profile or reinstall from Git.',
  desktopHalfPending: 'copying…',
  desktopHalfPendingTip:
    'This package ships a desktop half that has not been copied into the app yet. Use Rescan, or restart the app.',
  desktopHalfRemote: 'unavailable (remote backend)',
  desktopHalfRemoteTip:
    "This package's desktop half is on the remote backend's disk, which this app cannot read. To use it here, run Install from Git with the package's repo URL and the Desktop target checked — that clones the desktop half onto this machine.",
  emptyAll: 'No plugins yet.',
  empty: 'No agent plugins installed for this profile.',
  emptyHint: 'Browse the catalog below and install a reviewed plugin with one click.',
  loadFailed: 'Could not load agent plugins',
  toggleFailed: (name: string) => `Could not toggle ${name}`,
  toolsetOn: (name: string, profile: string) => `${name} agent tools enabled for ${profile}`,
  toolsetOff: (name: string, profile: string) => `${name} agent tools disabled for ${profile}`,
  toolsetToggleFailed: (name: string) =>
    `Could not toggle the ${name} agent tools; the Desktop panel was left unchanged`,
  legacyBackend: 'This backend predates key-addressed plugin toggles — update Hermes to manage it here.',
  portableBadge: 'portable',
  serverStates: {
    connected: 'connected',
    app_not_running: 'app not running',
    hermes_not_connected: 'MCP connection missing',
    endpoint_unavailable: 'endpoint unavailable',
    no_interactive_session: 'no interactive session',
    version_too_old: 'version too old',
    missing_app: 'app missing',
    unsupported_gpu: 'GPU not supported',
    unknown: 'status unknown'
  },
  catalogTitle: 'Plugin catalog',
  catalogBrowse: 'Browse',
  catalogHide: 'Hide the catalog browser',
  catalogHint:
    'Hit "+ Add to this Agent" on any plugin — reviewed entries install at their pinned commit into the selected profile. Bundled agent+desktop plugins offer both halves.',
  alreadyInstalled: (name: string) => `${name} is already installed in this profile.`,
  catalogProvenance: (sha: string) => `Installed from the Hermes catalog${sha ? ` at pin ${sha}` : ''}.`,
  pinnedProvenance: (sha: string) =>
    `Pinned to commit ${sha}. Updates are refused until it is reinstalled with a new pin.`,
  pinnedBadge: (sha: string) => `pinned @ ${sha}`,
  update: 'Update',
  updateToPin: (sha: string) => `Update to ${sha}`,
  updateFailed: (name: string) => `Could not update ${name}`,
  updated: (name: string) => `${name} updated. Restart the gateway to apply.`,
  upToDate: (name: string) => `${name} is already up to date.`,
  updateConsentTitle: (name: string) => `${name} needs your OK`,
  updateConsentBody: (name: string) =>
    `Updating ${name} brings changes you should review. Apply it only if you trust the source:`,
  updateConsentConfirm: 'Apply update',
  uninstall: 'Uninstall',
  uninstallTip: (name: string, profile: string) => `Uninstall ${name} from ${profile}`,
  uninstallConfirmTitle: (name: string) => `Uninstall ${name}?`,
  uninstallConfirmBody: (name: string, profile: string) =>
    `This deletes the plugin's files from the ${profile} profile. Any desktop half it shipped is removed with it. Reinstall it from the catalog or from Git at any time.`,
  uninstallFailed: (name: string) => `Could not uninstall ${name}`,
  uninstalled: (name: string) => `${name} uninstalled. Restart the gateway to unload it.`,
  uninstallDesktopTip: (name: string) => `Uninstall ${name} from this app`,
  uninstallDesktopConfirmBody: (name: string) =>
    `This deletes ${name} from the desktop-plugins folder on this computer and unloads it now. Reinstall it from Git or drop the folder back at any time.`,
  uninstalledDesktop: (name: string) => `${name} uninstalled.`,
  deepLinkErrorTitle: 'Plugin install link rejected',
  deepLinkCatalogInvalidName: 'The link\u2019s catalog name is missing or invalid.',
  deepLinkCatalogUnknown: (name: string) =>
    `\u201C${name}\u201D is not in the Hermes plugin catalog. Nothing was installed.`,
  deepLinkCatalogUnavailable:
    'Could not load the Hermes plugin catalog. Check your connection and open the link again.',
  settingsToggle: (name: string) => `Settings: ${name}`,
  settingsForm: {
    save: 'Save settings',
    saved: (name: string) => `${name} settings saved.`,
    saveFailed: (name: string) => `Could not save ${name} settings`,
    required: 'Required',
    secretSet: '•••••••• (set)',
    secretStoredAs: (env: string) =>
      `Stored in the profile's .env as ${env}, never in config.yaml; leave blank to keep the current value.`
  }
}

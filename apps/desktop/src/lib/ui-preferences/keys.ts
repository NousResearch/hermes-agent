/**
 * Which persisted keys are machine-level UI preferences (#145213).
 *
 * Only these leave localStorage for the machine's shared store, so native
 * Desktop and every browser that opens this machine's Webapp show the same
 * choices. Scoped families keep their scope inside the key name (a
 * `….profile.<name>` or per-connection suffix): only the location changes.
 *
 * Deliberately NOT here (they stay in this browser/window):
 * - credentials, grants and connection config: gateway registry, custom model
 *   endpoints, plugin decisions and preview-script grants, embed permissions;
 * - per-window or per-device runtime state: layout tree, open tiles and tabs,
 *   last route/session, scroll, zoom, pet and popout geometry, keep-awake;
 * - drafts, queues, journals and caches: composer queue, in-flight turns,
 *   pending/prepared submissions, previews, pricing, backend skin cache;
 * - seen/dismissed markers and onboarding progress.
 */

const SHARED_KEYS: ReadonlySet<string> = new Set([
  // Appearance.
  'hermes-desktop-theme-v2',
  'hermes-desktop-mode-v1',
  'hermes-desktop-profile-themes-v1',
  'hermes-desktop-profile-modes-v1',
  'hermes-desktop-user-themes-v1',
  'hermes.desktop.backdrop.v1',
  'hermes.desktop.chat-text-scale.v1',
  'hermes.desktop.interfaceMode.v1',
  'hermes.desktop.textDirection',
  'hermes.desktop.translucency.v2',
  'hermes.desktop.user-bubble-transparency.v1',
  // Sessions sidebar and profile rail.
  'hermes.desktop.agentsGroupedByWorkspace',
  'hermes.desktop.profileColors',
  'hermes.desktop.profileOrder',
  'hermes.desktop.profileRailVisible',
  'hermes.desktop.sessionListDensity',
  'hermes.desktop.sessionOrder',
  'hermes.desktop.showAllProfiles',
  'hermes.desktop.sidebarAgentsGrouped.allProfiles',
  'hermes.desktop.sidebarCardRows',
  'hermes.desktop.sidebarGrouping',
  'hermes.desktop.sidebarGrouping.allProfiles',
  'hermes.desktop.sidebarRowMeta',
  'hermes.desktop.sidebarShowAllSessions',
  'hermes.desktop.sidebarShowArchived',
  'hermes.desktop.sidebarSortKey',
  // Status bar.
  'hermes.desktop.statusbarHidden.v2',
  'hermes.desktop.statusbarVisible.v2',
  // Transcript and tools.
  'hermes.desktop.hideThreadTimeline',
  'hermes.desktop.reasoning.collapsedByDefault',
  'hermes.desktop.toolView.hideCodeDiffs',
  'hermes.desktop.toolView.technical',
  // Shell, panes and behaviour.
  'hermes.desktop.alwaysExternalLinks.v1',
  'hermes.desktop.capabilities.catalogCardView',
  'hermes.desktop.composerPopout.enabled',
  'hermes.desktop.composerPopout.gesturesEnabled',
  'hermes.desktop.files.showIgnored',
  'hermes.desktop.keybinds',
  'hermes.desktop.layoutPresets.v2',
  'hermes.desktop.panesFlipped',
  'hermes.desktop.tabStripDefault',
  'hermes.desktop.tips.rotation.v1',
  'hermes.desktop.titlebarAppActions',
  'hermes.desktop.tours.v1',
  // Sound and media.
  'hermes.desktop.autoSpeakReplies',
  'hermes.desktop.completionSoundVariantId',
  'hermes.desktop.hapticsMuted',
  'hermes.desktop.videoPlaybackSpeed',
  // Model picker.
  'hermes.desktop.collapsed-providers',
  'hermes.desktop.favorite-models',
  'hermes.desktop.model-presets',
  'hermes.desktop.visible-models'
])

/** Families whose scope rides in the key suffix (gateway aliases/order/collapsed). */
const SHARED_PREFIXES: readonly string[] = ['hermes.desktop.sidebar.gatewayGroups.v1.']

export function isSharedPreferenceKey(key: string): boolean {
  return SHARED_KEYS.has(key) || SHARED_PREFIXES.some(prefix => key.startsWith(prefix))
}

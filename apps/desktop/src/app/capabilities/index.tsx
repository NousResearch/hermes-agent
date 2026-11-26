import './ui/page.css'

import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { ProfileScope } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $gateway, activeGatewayConnectionId } from '@/store/gateway'
import { OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $capabilitiesManagedPerAgent } from '@/store/shell-policy'
import { usePortalShell } from '@/store/shell-variant'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PanelEmpty } from '../overlays/panel'
import { PageSearchShell, type PageShellTab } from '../page-search-shell'
import { SETTINGS_ROUTE } from '../routes'
import { useSettingsOwner } from '../settings/profile-scope'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { type CapabilityCount, useConnectorsCount, usePluginsCount } from './capability-counts'
import { syncCatalogArcs } from './catalog/catalog-arc'
import { prefetchCatalogWhenIdle } from './catalog/catalog-data'
import { CatalogSkeleton } from './catalog/catalog-skeleton'
import { ConnectorsTab } from './connectors/connectors-tab'
import { type ConnectionsCopy, useConnectionsCopy } from './connectors/redesign/connections-copy'
import { useCapabilitiesPageDescription } from './page-copy'
import { PluginsTab } from './plugins/plugins-tab'
import {
  type CapabilityScope,
  CapabilityScopeSelector,
  profileScopeConnectionId,
  useCapabilityScope
} from './scope-selector'
import { SKILLS_QUERY_KEY, useSkillsQuery } from './skills/skills-data'
import { SkillsTab } from './skills/skills-tab'
import { AgentToolsets } from './toolsets/agent-toolsets'
import { refreshToolCalls } from './toolsets/tool-calls'
import { TOOLSETS_QUERY_KEY, toolsetSearchTerms, useToolsetsQuery, visibleToolsetCount } from './toolsets/toolsets-data'
import { ToolsetsTab } from './toolsets/toolsets-tab'
import { AgentCapabilitySkeleton } from './ui/agent-capability'
import { CapabilitySearch } from './ui/capability-search'

// Skills Hub browsing lives inside the Skills tab. Legacy `?tab=hub`
// links fall back to 'skills' via useRouteEnumParam.
const CAPABILITY_MODES = ['skills', 'plugins', 'connectors', 'toolsets'] as const

type CapabilityMode = (typeof CAPABILITY_MODES)[number]

// The redesign's page leads with what Hermes is connected to, then what extends it, then what it
// knows how to do; it opens on the first. Classic keeps its own order above. No Tools tab: here it
// was a list with nothing to configure, and an agent's tools are switched in that agent's settings
// (which pins this view to `toolsets`). A link to the old tab lands on the first.
const PORTAL_CAPABILITY_MODES = ['connectors', 'plugins', 'skills'] as const satisfies readonly CapabilityMode[]

interface CapabilitiesViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
  /** Embedded mode (plugin dialogs — e.g. Bot Mode's Advanced section): tab
   *  state lives in local React state instead of the route's `?tab=` param,
   *  so an embedding dialog never fights the page router. */
  embedded?: boolean
  /** Agent settings owns the navigation when a single capability is embedded. */
  fixedMode?: CapabilityMode
  /** The agent's own settings page: its Plugins list is what the agent has installed, with no
   *  library and no way to install. Dialogs that pin an agent do not ask for this. */
  installedPluginsOnly?: boolean
  /** Pin the WHOLE view to one profile: the scope selector is hidden and
   *  every tab reads/writes THAT profile. This is the plugin door — Bot Mode
   *  renders the real Capabilities surface pinned to a bot. */
  fixedProfile?: string
  /** Pin the view to a REGISTERED gateway connection alongside `fixedProfile`:
   *  every read/write routes to that machine's backend instead of the active
   *  gateway. `''`/`'local'` mean the local pool. This is Bot Mode's
   *  remote-target door — a bot living on another registered gateway gets the
   *  live surface pointed at ITS backend. Ignored without `fixedProfile`. */
  fixedConnection?: string
}

/** The Capabilities page SHELL: tab selection, the search header, the profile /
 *  connection scope, the refresh hotkey, and the dispatch to one tab component
 *  per tab. Each tab owns its list, detail pane and writes; the two installed
 *  lists are fetched here because the tab pills count them for the tab the user
 *  is NOT on. */
export function CapabilitiesView({
  embedded = false,
  installedPluginsOnly = false,
  fixedConnection,
  fixedProfile,
  fixedMode,
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: CapabilitiesViewProps) {
  const { t } = useI18n()
  const redesign = usePortalShell()
  const presentation = capabilitiesPresentation(redesign, embedded, Boolean(fixedProfile), installedPluginsOnly)
  const agentControls = !useStore($capabilitiesManagedPerAgent) || Boolean(fixedProfile)
  const modes = capabilityModes(redesign, Boolean(fixedProfile))
  const [selectedMode, setMode] = useSelectedMode(modes, embedded)
  const mode = fixedMode ?? selectedMode
  const pageDescription = useCapabilitiesPageDescription(mode === 'connectors')
  const connCopy = useConnectionsCopy()
  const gateway = useStoreSelector($gateway, g => (mode === 'connectors' ? g : null))
  const reloadsOwnerChat = useReloadsOwnerChat()

  const [query, setQuery] = useState('')
  const [connectorsSubview, setConnectorsSubview] = useState(false)

  // A drill-down belongs to the tab it was opened on; leaving Connectors forgets it.
  useEffect(() => {
    if (mode !== 'connectors') {
      setConnectorsSubview(false)
    }
  }, [mode])

  const scope = useCapabilityScope({ fixedConnection, fixedProfile })

  const pluginsReachable = usePluginsReachable(scope)

  const { openAgentSettings } = useCapabilityAgentSettings(scope, mode)

  // The two installed lists the tab pills count. They are fetched here, as a
  // pair, because the counts stay live for the tab the user is NOT on.
  const { data: skills, isError: skillsFailed, error: skillsError } = useSkillsQuery(scope.profile)
  const { data: toolsets, isError: toolsetsFailed } = useToolsetsQuery(scope.profile)

  const refreshCapabilities = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: SKILLS_QUERY_KEY }),
      queryClient.invalidateQueries({ queryKey: TOOLSETS_QUERY_KEY }),
      queryClient.invalidateQueries({ queryKey: OFFICIAL_SKILLS_KEY })
    ])

    invalidateSlashCompletions()

    void refreshToolCalls(scope.profile)
  }, [scope.profile])

  useRefreshHotkey(refreshCapabilities)

  // Plugins is small enough to warm from any tab. Skills (~100k rows) only
  // loads when asked for: an idle parse of it would still block the page.
  usePluginCatalogWarmup(mode, presentation.installedPlugins)
  useEffect(syncCatalogArcs, [])

  // Rotating placeholder nudges from the user's own data — teach that search
  // understands tool names, not just titles.
  const toolsetHints = useMemo(
    () => (toolsets?.length ? toolsetSearchTerms(toolsets).map(term => t.common.tryHint(term)) : undefined),
    [t, toolsets]
  )

  // MCP and Plugins load independently of the installed Skills/Tools lists.
  const pending = mode === 'toolsets' && !(skills && toolsets)
  const redesignedInventory = presentation.inventoryFrame

  const tabContent = {
    // The gateway instance backs ONLY the live `reload.mcp` RPC, and it is the
    // ACTIVE gateway's socket — for a scope pinned to a different backend that
    // (config edits still apply on that backend's next session).
    connectors: () => (
      // Connectors are switched here in both shells: in the redesign this page is the only place
      // they are managed (an agent's settings have no Connectors page), for the agent in its header.
      <ConnectorsTab
        agentControls={!scope.resolving}
        agentLayout={presentation.agentLayout}
        gateway={scope.crossBackend || !reloadsOwnerChat ? null : gateway}
        key={`connectors-${scope.key}`}
        onSubviewChange={setConnectorsSubview}
        ownerLabel={inventoryOwnerLabel(presentation, scope)}
        profile={scope.profile}
        redesignLayout={redesignedInventory}
      />
    ),
    plugins: () =>
      redesign && !pluginsReachable ? (
        <PluginsElsewhere onOpenAgentSettings={openAgentSettings} />
      ) : (
        <PluginsTab
          agentControls={agentControls}
          agentLayout={presentation.agentLayout}
          installedOnly={presentation.installedPlugins}
          key={`plugins-${scope.key}`}
          onQueryChange={setQuery}
          profile={scope.profile}
          query={query}
          scopeLabel={scope.label}
          scopeSelector={pluginsScopeSelector(agentControls, scope)}
        />
      ),
    skills: () => (
      <SkillsTab
        agentControls={agentControls}
        agentLayout={presentation.agentLayout}
        installedError={skillsError}
        installedPending={!skills || skillsFailed}
        key={`skills-${scope.key}`}
        onQueryChange={setQuery}
        onRefresh={() => void refreshCapabilities()}
        profile={scope.profile}
        query={query}
        skills={skills ?? []}
      />
    ),
    toolsets: () =>
      presentation.agentLayout ? (
        <AgentToolsets
          key={`toolsets-${scope.key}`}
          onQueryChange={setQuery}
          profile={scope.profile}
          query={query}
          toolsets={toolsets ?? []}
        />
      ) : (
        <ToolsetsTab
          hints={toolsetHints}
          key={`toolsets-${scope.key}`}
          onQueryChange={setQuery}
          profile={scope.profile}
          query={query}
          toolsets={toolsets ?? []}
        />
      )
  } satisfies Record<CapabilityMode, () => React.ReactNode>

  // The redesigned inventory names Connectors "Connections"; its own page owns the title, and a
  // category See-All or an app's Tools page owns the header entirely while open.
  const { drilldown: connectorsDrilldown, frame: connectorsFrame } = connectorsShellState(
    redesignedInventory,
    mode,
    connectorsSubview
  )
  const meta = capabilitiesShellMeta({
    connCopy,
    connectorsDrilldown,
    connectorsFrame,
    connectorsLabel: redesignedInventory ? connCopy.page.title : undefined,
    description: pageDescription,
    fixedMode,
    modes,
    presentation,
    skills,
    t,
    toolsets
  })

  const shellProps: React.ComponentProps<typeof PageSearchShell> & { 'data-capabilities-v2'?: true } = {
    ...props,
    activeTab: mode,
    'data-capabilities-v2': presentation.pageFrame || undefined,
    hideHeader: meta.hideHeader,
    onTabChange: id => setMode(id as CapabilityMode),
    pageDescription: meta.pageDescription,
    pageTitle: meta.pageTitle,
    // Every tab renders the shared CapabilitySearch beside its own results.
    searchHidden: true,
    searchTrailingAction: capabilitiesHeaderAction(
      presentation,
      mode,
      scope,
      agentControls,
      openAgentSettings
    ),
    tabs: meta.tabs,
    tabsAlign: 'start',
    children: (
      <CapabilitiesBody agentLayout={presentation.agentLayout}>
        <CapabilityLoadGate
          error={skillsError}
          failed={skillsFailed || toolsetsFailed}
          onRefresh={refreshCapabilities}
          pending={pending}
          skeleton={capabilitiesSkeleton(presentation, query, setQuery)}
        >
          {tabContent[mode]()}
        </CapabilityLoadGate>
      </CapabilitiesBody>
    )
  }

  return presentation.pageFrame ? (
    <CountedCapabilitiesShell {...shellProps} pluginsReachable={pluginsReachable} scope={scope.profile} />
  ) : (
    <PageSearchShell {...shellProps} />
  )
}

interface CountedCapabilitiesShellProps extends React.ComponentProps<typeof PageSearchShell> {
  scope: ProfileScope
  pluginsReachable: boolean
}

/** The redesigned page also counts Plugins and Connectors on their tabs (919:9095); `null` reserves a
 *  count still loading. Their reads mount with this frame only, so classic and embedded views,
 *  which show no such counts, never fetch them. */
function CountedCapabilitiesShell({ pluginsReachable, scope, tabs, ...props }: CountedCapabilitiesShellProps) {
  const plugins = usePluginsCount(scope, pluginsReachable)
  const connectors = useConnectorsCount(scope)
  const counts: Partial<Record<string, CapabilityCount>> = { connectors, ...(pluginsReachable && { plugins }) }

  // A count the agent's backend could not answer drops its slot rather than reserving it forever.
  const counted = tabs?.map((tab): PageShellTab => {
    const count = counts[tab.id]

    return count ? { ...tab, meta: count.failed ? undefined : (count.value ?? null) } : tab
  })

  return <PageSearchShell {...props} tabs={counted} />
}

interface ToolsetsSkeletonProps {
  agentLayout: boolean
  query: string
  onQueryChange: (query: string) => void
}

/** The Tools tab's first paint while its lists load (redesign): an agent's sections, or the library's
 *  live search over its shelves. */
function ToolsetsSkeleton({ agentLayout, query, onQueryChange }: ToolsetsSkeletonProps) {
  const { t } = useI18n()

  if (agentLayout) {
    return <AgentCapabilitySkeleton />
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="catalog-search-header shrink-0">
        <CapabilitySearch
          onChange={onQueryChange}
          placeholder={t.skills.searchToolsets}
          scope={t.skills.tabToolsets}
          value={query}
        />
      </header>
      <CatalogSkeleton />
    </div>
  )
}

interface CapabilitiesPresentation {
  /** The redesigned page frame (catalog hero art, v2 card layout). */
  pageFrame: boolean
  /** A routed page with its own H1 rather than a surface embedded in another page. */
  pageTitle: boolean
  /** One agent's enablement editor (agent settings, Bot Mode) rather than the inventory. */
  agentLayout: boolean
  /** The redesigned inventory proper: the page frame, not an agent's editor. */
  inventoryFrame: boolean
  /** The agent's own settings page lists only the plugins it has installed, and says so itself
   *  (`installedPluginsOnly`). A dialog that pins the same layout (creating or editing a bot) does
   *  not, and keeps the library and its ways in. */
  installedPlugins: boolean
  /** Classic keeps the scope selector beside the header search. */
  headerScope: boolean
}

/** One place derives how this view presents itself from the shell, embedding and pinning. */
function capabilitiesPresentation(
  redesign: boolean,
  embedded: boolean,
  pinned: boolean,
  installedPluginsOnly: boolean
): CapabilitiesPresentation {
  const pageFrame = redesign && !embedded
  const agentLayout = redesign && pinned

  return {
    pageFrame,
    pageTitle: !embedded && !(redesign && pinned),
    agentLayout,
    inventoryFrame: pageFrame && !agentLayout,
    installedPlugins: agentLayout && installedPluginsOnly,
    headerScope: !redesign
  }
}

// Feature-detection flag for plugins (Bot Mode): TRUE means this build's
// CapabilitiesView routes `fixedConnection` to the pinned connection's backend.
// Older builds export CapabilitiesView WITHOUT the prop — passing it there would
// silently read/write the ACTIVE gateway under the remote bot's profile name,
// which is exactly the wrong-machine bug the prop exists to prevent. A static
// property is probe-able without rendering.
CapabilitiesView.supportsFixedConnection = true as const

interface CapabilityLoadGateProps {
  /** The redesign's first paint while the lists load: the tab's frame, not a spinner. */
  skeleton?: React.ReactNode
  pending: boolean
  failed: boolean
  error: unknown
  onRefresh: () => Promise<void>
  children: React.ReactNode
}

function CapabilityLoadGate({ pending, failed, error, onRefresh, skeleton, children }: CapabilityLoadGateProps) {
  const { t } = useI18n()

  if (!pending) {
    return children
  }

  if (!failed) {
    return skeleton ?? <PageLoader label={t.skills.loading} />
  }

  return (
    <PanelEmpty
      action={
        <Button onClick={() => void onRefresh()} size="sm">
          {t.skills.refresh}
        </Button>
      }
      description={error instanceof Error ? error.message : undefined}
      icon="error"
      title={t.skills.skillsLoadFailed}
    />
  )
}

/** Keep the inventory's hand-off pinned to the same owner as its queries. */
/** The redesign's stand-in for a scope whose plugins this window cannot reach. Classic always
 *  lists them, as it always has. */
function PluginsElsewhere({ onOpenAgentSettings }: { onOpenAgentSettings: () => void }) {
  const { t } = useI18n()

  return (
    <div className="grid justify-items-start gap-3 p-5">
      <p>{t.skills.remotePluginHint}</p>
      <Button onClick={onOpenAgentSettings}>{t.skills.openAgentSettings}</Button>
    </div>
  )
}

/** The inventory's header owns the scope; an agent's own Plugins column names it. */
function pluginsScopeSelector(agentControls: boolean, scope: CapabilityScope) {
  return agentControls && scope.options.length > 1 ? <CapabilityScopeSelector compact scope={scope} /> : undefined
}

/** `reload.mcp` reloads the window's chat, which is the ACTIVE agent's: an agent page showing
 *  another agent leaves its MCP edits to that agent's next session instead. */
function useReloadsOwnerChat(): boolean {
  const owner = useSettingsOwner()
  const activeProfile = useStore($activeGatewayProfile)

  return !owner || normalizeProfileKey(owner.profile) === normalizeProfileKey(activeProfile)
}

/** The modes the page addresses. An agent's own settings pin this view to that agent and address
 *  every mode, Tools included. */
const capabilityModes = (redesign: boolean, pinned: boolean): readonly CapabilityMode[] =>
  redesign && !pinned ? PORTAL_CAPABILITY_MODES : CAPABILITY_MODES

/** The Connectors tab's drill-down owns the page header; both the frame and its state derive here. */
function connectorsShellState(
  inventoryFrame: boolean,
  mode: CapabilityMode,
  subview: boolean
): { drilldown: boolean; frame: boolean } {
  const frame = inventoryFrame && mode === 'connectors'

  return { drilldown: frame && subview, frame }
}

interface CapabilitiesShellMeta {
  hideHeader: boolean
  pageDescription?: string
  pageTitle?: string
  tabs: PageShellTab[]
}

interface CapabilitiesShellMetaInput {
  connCopy: ConnectionsCopy
  connectorsDrilldown: boolean
  connectorsFrame: boolean
  connectorsLabel?: string
  description?: string
  fixedMode?: CapabilityMode
  modes: readonly CapabilityMode[]
  presentation: CapabilitiesPresentation
  skills?: unknown[]
  t: Translations
  toolsets?: Parameters<typeof visibleToolsetCount>[0]
}

/** The shell's own title, description and tabs: a drill-down hides the header (its surface draws the
 *  back link), the inventory names Connections, and a pinned agent editor drops the tabs. */
function capabilitiesShellMeta({
  connCopy,
  connectorsDrilldown,
  connectorsFrame,
  connectorsLabel,
  description,
  fixedMode,
  modes,
  presentation,
  skills,
  t,
  toolsets
}: CapabilitiesShellMetaInput): CapabilitiesShellMeta {
  const tabs =
    fixedMode || connectorsDrilldown ? [] : capabilityTabs(modes, t, skills, toolsets, connectorsLabel)

  if (connectorsFrame) {
    return {
      hideHeader: connectorsDrilldown,
      pageDescription: connectorsDrilldown ? undefined : connCopy.page.subtitle,
      pageTitle: connCopy.page.title,
      tabs
    }
  }

  return {
    hideHeader: false,
    pageDescription: presentation.pageTitle ? description : undefined,
    pageTitle: presentation.pageTitle ? t.sidebar.nav.capabilities : undefined,
    tabs
  }
}

/** Classic keeps the scope selector beside the search; the inventory hands off to agent settings. */
function capabilitiesHeaderAction(
  presentation: CapabilitiesPresentation,
  mode: CapabilityMode,
  scope: CapabilityScope,
  agentControls: boolean,
  onOpenAgentSettings: () => void
): React.ReactNode {
  if (presentation.headerScope) {
    return mode === 'plugins' ? undefined : <CapabilityScopeSelector scope={scope} />
  }

  if (agentControls) {
    return undefined
  }

  return (
    <CapabilityInventoryScope
      managedHere={mode === 'connectors'}
      onOpenAgentSettings={onOpenAgentSettings}
      scope={scope}
    />
  )
}

/** An agent editor scrolls as one page, so its capability list sizes to content. */
function CapabilitiesBody({ agentLayout, children }: { agentLayout: boolean; children: React.ReactNode }) {
  return (
    <div className={agentLayout ? 'grid' : 'flex h-full flex-col'}>
      <div className={agentLayout ? 'grid' : 'flex min-h-0 flex-1 flex-col'}>{children}</div>
    </div>
  )
}

function capabilitiesSkeleton(
  presentation: CapabilitiesPresentation,
  query: string,
  onQueryChange: (query: string) => void
): React.ReactNode {
  if (!presentation.agentLayout && !presentation.pageFrame) {
    return undefined
  }

  return <ToolsetsSkeleton agentLayout={presentation.agentLayout} onQueryChange={onQueryChange} query={query} />
}

/** Plugins is small enough to warm from any other tab. An agent's own pages list only what it has
 *  installed: nothing there reads the library. */
function usePluginCatalogWarmup(mode: CapabilityMode, installedOnly: boolean) {
  const warms = mode !== 'plugins' && !installedOnly

  useEffect(() => (warms ? prefetchCatalogWhenIdle('plugins') : undefined), [warms])
}

/** The tab the page is on, and its setter. Both hooks run unconditionally (rules of hooks); embedded
 *  picks the local one so tab clicks inside a dialog don't rewrite the page URL. */
function useSelectedMode(modes: readonly CapabilityMode[], embedded?: boolean) {
  const routeTab = useRouteEnumParam('tab', modes, modes[0])
  const localTab = useState<CapabilityMode>(modes[0])

  return embedded ? localTab : routeTab
}

/** The page's tabs; the two installed lists count on theirs (`null` reserves a count still loading). */
function capabilityTabs(
  modes: readonly CapabilityMode[],
  t: Translations,
  skills?: unknown[],
  toolsets?: Parameters<typeof visibleToolsetCount>[0],
  /** The redesign calls the surface Connections; classic keeps Connectors. */
  connectorsLabel?: string
): PageShellTab[] {
  const tabs: Record<CapabilityMode, Omit<PageShellTab, 'id'>> = {
    skills: { label: t.skills.tabSkills, meta: skills?.length ?? null },
    plugins: { label: t.skills.tabPlugins },
    connectors: { label: connectorsLabel ?? t.connectorsPage.title },
    toolsets: { label: t.skills.tabToolsets, meta: toolsets ? visibleToolsetCount(toolsets) : null }
  }

  return modes.map(id => ({ id, ...tabs[id] }))
}

/** Plugins rides the settings request: the active gateway, or inside an agent's own settings that
 *  agent's backend. A scope on any other machine is out of its reach. */
function usePluginsReachable(scope: CapabilityScope): boolean {
  const owner = useSettingsOwner()

  return (
    !scope.crossBackend ||
    (owner !== undefined && (owner.connectionId ?? 'local') === profileScopeConnectionId(scope.profile))
  )
}

function useCapabilityAgentSettings(scope: CapabilityScope, mode: CapabilityMode) {
  const navigate = useNavigate()

  const ownerProfile =
    typeof scope.profile === 'object' ? (scope.profile?.profile ?? 'default') : (scope.profile ?? 'default')

  const ownerConnection =
    (typeof scope.profile === 'object' ? scope.profile?.connectionId : undefined) ??
    activeGatewayConnectionId() ??
    'local'

  const ownerQuery = new URLSearchParams({
    agentProfile: ownerProfile,
    panel: 'capabilities',
    tab: mode,
    connection: ownerConnection
  })

  return {
    openAgentSettings: () => navigate(`${SETTINGS_ROUTE}?${ownerQuery}`)
  }
}

interface CapabilityInventoryScopeProps {
  scope: CapabilityScope
  onOpenAgentSettings: () => void
  /** The tab's switches are on this page (Connectors): there is nothing to hand off for. */
  managedHere: boolean
}

/** Whose connectors the redesign's page is editing, for the tab to say beside its lists. */
function inventoryOwnerLabel(presentation: CapabilitiesPresentation, scope: CapabilityScope) {
  const profile = typeof scope.profile === 'object' ? scope.profile?.profile : scope.profile

  // The roster's name for it, or the profile's own while the roster loads.
  return presentation.pageFrame ? (scope.label ?? profile ?? undefined) : undefined
}

/** The inventory's owner, in the header's trailing slot where 919:9095 puts its one quiet link: which
 *  agent and machine the page reads, and the hand-off to that agent's settings, where switches live.
 *  The header owns the slot's height, so the select resolving its options never moves the page. */
function CapabilityInventoryScope({ scope, onOpenAgentSettings, managedHere }: CapabilityInventoryScopeProps) {
  const { t } = useI18n()

  return (
    <div className="capability-header-scope" data-slot="capability-scope">
      <CapabilityScopeSelector quiet scope={scope} />
      {!managedHere && (
        <Tip label={t.skills.inventoryScopeHint}>
          <Button className="capability-header-link" onClick={onOpenAgentSettings} size="inline" variant="text">
            {t.skills.openAgentSettings}
            <Codicon aria-hidden name="arrow-right" size="0.75rem" />
          </Button>
        </Tip>
      )}
    </div>
  )
}

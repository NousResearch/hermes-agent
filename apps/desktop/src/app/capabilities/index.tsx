import './ui/page.css'

import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $gateway, activeGatewayConnectionId } from '@/store/gateway'
import { OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'
import { $simplifiedAgentMode } from '@/store/simplified-agent-mode'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PanelEmpty } from '../overlays/panel'
import { PageSearchShell } from '../page-search-shell'
import { SETTINGS_ROUTE } from '../routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { syncCatalogArcs } from './catalog/catalog-arc'
import { prefetchCatalogWhenIdle } from './catalog/catalog-data'
import { ConnectorsTab } from './connectors/connectors-tab'
import { PluginsTab } from './plugins/plugins-tab'
import { type CapabilityScope, CapabilityScopeSelector, useCapabilityScope } from './scope-selector'
import { SKILLS_QUERY_KEY, useSkillsQuery } from './skills/skills-data'
import { SkillsTab } from './skills/skills-tab'
import { AgentToolsets } from './toolsets/agent-toolsets'
import { refreshToolCalls } from './toolsets/tool-calls'
import { TOOLSETS_QUERY_KEY, toolsetSearchTerms, useToolsetsQuery, visibleToolsetCount } from './toolsets/toolsets-data'
import { ToolsetsTab } from './toolsets/toolsets-tab'

// Skills Hub browsing lives inside the Skills tab. Legacy `?tab=hub`
// links fall back to 'skills' via useRouteEnumParam.
const CAPABILITY_MODES = ['skills', 'plugins', 'connectors', 'toolsets'] as const

type CapabilityMode = (typeof CAPABILITY_MODES)[number]

interface CapabilitiesViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
  /** Embedded mode (plugin dialogs — e.g. Bot Mode's Advanced section): tab
   *  state lives in local React state instead of the route's `?tab=` param,
   *  so an embedding dialog never fights the page router. */
  embedded?: boolean
  /** Agent settings owns the navigation when a single capability is embedded. */
  fixedMode?: CapabilityMode
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
  fixedConnection,
  fixedProfile,
  fixedMode,
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: CapabilitiesViewProps) {
  const { t } = useI18n()
  const redesignEnabled = useStore($simplifiedAgentMode)
  const agentRedesign = redesignEnabled && Boolean(fixedProfile)
  const agentControls = !redesignEnabled || Boolean(fixedProfile)
  const redesigned = redesignEnabled && !embedded
  // Both hooks run unconditionally (rules of hooks); embedded picks the local
  // one so tab clicks inside a dialog don't rewrite the page URL.
  const routeTab = useRouteEnumParam('tab', CAPABILITY_MODES, 'skills')
  const localTab = useState<CapabilityMode>('skills')
  const [selectedMode, setMode] = embedded ? localTab : routeTab
  const mode = fixedMode ?? selectedMode
  const gateway = useStoreSelector($gateway, g => (mode === 'connectors' ? g : null))

  const [query, setQuery] = useState('')

  const scope = useCapabilityScope({ fixedConnection, fixedProfile })

  const { ownerProfile, ownerConnection, openAgentSettings } = useCapabilityAgentSettings(scope, mode)

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
  useEffect(() => (mode === 'plugins' ? undefined : prefetchCatalogWhenIdle('plugins')), [mode])
  useEffect(syncCatalogArcs, [])

  // Rotating placeholder nudges from the user's own data — teach that search
  // understands tool names, not just titles.
  const toolsetHints = useMemo(
    () => (toolsets?.length ? toolsetSearchTerms(toolsets).map(term => t.common.tryHint(term)) : undefined),
    [t, toolsets]
  )

  // MCP and Plugins load independently of the installed Skills/Tools lists.
  const gated = mode === 'toolsets'
  const pending = gated && !(skills && toolsets)

  const tabContent = {
    // The gateway instance backs ONLY the live `reload.mcp` RPC, and it is the
    // ACTIVE gateway's socket — for a scope pinned to a different backend that
    // (config edits still apply on that backend's next session).
    connectors: () => (
      <ConnectorsTab
        agentControls={agentControls}
        agentLayout={agentRedesign}
        gateway={scope.crossBackend ? null : gateway}
        key={`connectors-${scope.key}`}
        profile={scope.profile}
      />
    ),
    // Agent plugins for the scoped profile and app-level desktop plugins.
    plugins: () =>
      scope.crossBackend ? (
        <div className="grid justify-items-start gap-3 p-5">
          <p>{t.skills.remotePluginHint}</p>
          <Button onClick={openAgentSettings}>{t.skills.openAgentSettings}</Button>
        </div>
      ) : (
        <PluginsTab
          agentControls={agentControls}
          agentLayout={agentRedesign}
          key={`plugins-${scope.key}`}
          onQueryChange={setQuery}
          profile={scope.profile}
          query={query}
          scopeLabel={scope.label}
          scopeSelector={scope.options.length > 1 ? <CapabilityScopeSelector compact scope={scope} /> : undefined}
        />
      ),
    skills: () => (
      <SkillsTab
        agentControls={agentControls}
        agentLayout={agentRedesign}
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
      agentRedesign ? (
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

  return (
    <PageSearchShell
      {...props}
      activeTab={mode}
      data-capabilities-v2={redesigned || undefined}
      onTabChange={id => setMode(id as CapabilityMode)}
      pageTitle={embedded || agentRedesign ? undefined : t.sidebar.nav.capabilities}
      // Every tab renders the shared CapabilitySearch beside its own results.
      searchHidden
      searchTrailingAction={!redesignEnabled && mode !== 'plugins' && <CapabilityScopeSelector scope={scope} />}
      tabs={
        fixedMode
          ? []
          : [
              { id: 'skills', label: t.skills.tabSkills, meta: skills?.length ?? null },
              { id: 'plugins', label: t.skills.tabPlugins },
              { id: 'connectors', label: t.connectorsPage.title },
              { id: 'toolsets', label: t.skills.tabToolsets, meta: toolsets ? visibleToolsetCount(toolsets) : null }
            ]
      }
      tabsAlign="start"
    >
      <div className="flex h-full flex-col">
        {!agentControls && (
          <CapabilityInventoryScope
            onOpenAgentSettings={openAgentSettings}
            ownerConnection={ownerConnection}
            ownerProfile={ownerProfile}
            scope={scope}
          />
        )}
        <div className="flex min-h-0 flex-1 flex-col">
          <CapabilityLoadGate
            error={skillsError}
            failed={skillsFailed || toolsetsFailed}
            onRefresh={refreshCapabilities}
            pending={pending}
          >
            {tabContent[mode]()}
          </CapabilityLoadGate>
        </div>
      </div>
    </PageSearchShell>
  )
}

// Feature-detection flag for plugins (Bot Mode): TRUE means this build's
// CapabilitiesView routes `fixedConnection` to the pinned connection's backend.
// Older builds export CapabilitiesView WITHOUT the prop — passing it there would
// silently read/write the ACTIVE gateway under the remote bot's profile name,
// which is exactly the wrong-machine bug the prop exists to prevent. A static
// property is probe-able without rendering.
CapabilitiesView.supportsFixedConnection = true as const

interface CapabilityLoadGateProps {
  pending: boolean
  failed: boolean
  error: unknown
  onRefresh: () => Promise<void>
  children: React.ReactNode
}

function CapabilityLoadGate({ pending, failed, error, onRefresh, children }: CapabilityLoadGateProps) {
  const { t } = useI18n()

  if (!pending) {
    return children
  }

  if (!failed) {
    return <PageLoader label={t.skills.loading} />
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
function useCapabilityAgentSettings(scope: CapabilityScope, mode: CapabilityMode) {
  const navigate = useNavigate()

  const ownerProfile =
    typeof scope.profile === 'object' ? (scope.profile?.profile ?? 'default') : (scope.profile ?? 'default')

  const ownerConnection =
    (typeof scope.profile === 'object' ? scope.profile?.connectionId : undefined) ??
    activeGatewayConnectionId() ??
    'local'

  const ownerQuery = new URLSearchParams({ agentProfile: ownerProfile, panel: 'capabilities', tab: mode })

  if (ownerConnection) {
    ownerQuery.set('connection', ownerConnection)
  }

  return {
    ownerProfile,
    ownerConnection,
    openAgentSettings: () => navigate(`${SETTINGS_ROUTE}?${ownerQuery}`)
  }
}

interface CapabilityInventoryScopeProps {
  scope: CapabilityScope
  ownerProfile: string
  ownerConnection: string
  onOpenAgentSettings: () => void
}

/** Inventory browsing explains its owner without exposing agent enablement switches. */
function CapabilityInventoryScope({
  scope,
  ownerProfile,
  ownerConnection,
  onOpenAgentSettings
}: CapabilityInventoryScopeProps) {
  const { t } = useI18n()

  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-(--ui-stroke-tertiary) px-5 py-3">
      <div className="min-w-0 flex-1 text-xs text-(--ui-text-tertiary)">
        <p>{t.skills.inventoryScopeHint}</p>
        <p className="mt-1">
          {t.skills.configuringProfile} {scope.label ?? ownerProfile}
          {ownerConnection ? ` · ${ownerConnection}` : ''}
        </p>
      </div>
      <CapabilityScopeSelector compact scope={scope} />
      <Button onClick={onOpenAgentSettings} size="sm" variant="outline">
        {t.skills.openAgentSettings}
      </Button>
    </div>
  )
}

import type * as React from 'react'
import { useCallback, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $gateway } from '@/store/gateway'
import { OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PanelEmpty } from '../overlays/panel'
import { PageSearchShell } from '../page-search-shell'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { CapabilityTabs, type CapabilityView } from './capability-tabs'
import { ConnectorsTab } from './connectors/connectors-tab'
import { PluginsTab } from './plugins/plugins-tab'
import { CapabilityScopeSelector, useCapabilityScope } from './scope-selector'
import { SkillCatalog } from './skill-catalog'
import { EmbeddedHubPicker } from './skills/embedded-hub-picker'
import { SKILLS_QUERY_KEY, skillSearchTerms, useSkillsQuery } from './skills/skills-data'
import { SkillsTab } from './skills/skills-tab'
import { refreshToolCalls } from './toolsets/tool-calls'
import { TOOLSETS_QUERY_KEY, toolsetSearchTerms, useToolsetsQuery, visibleToolsetCount } from './toolsets/toolsets-data'
import { ToolsetsTab } from './toolsets/toolsets-tab'
import { UpdateSkillsButton } from './update-skills-button'

// Native catalog browsing and the full Hub each retain their own surface.
const CAPABILITY_MODES = ['skills', 'toolsets', 'connectors', 'plugins', 'hub'] as const

type CapabilityMode = (typeof CAPABILITY_MODES)[number]

interface CapabilitiesViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
  /** Embedded mode (plugin dialogs — e.g. Bot Mode's Advanced section): tab
   *  state lives in local React state instead of the route's `?tab=` param,
   *  so an embedding dialog never fights the page router. */
  embedded?: boolean
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
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: CapabilitiesViewProps) {
  const { t } = useI18n()
  // Both hooks run unconditionally (rules of hooks); embedded picks the local
  // one so tab clicks inside a dialog don't rewrite the page URL.
  const routeTab = useRouteEnumParam('tab', CAPABILITY_MODES, 'skills')
  const localTab = useState<CapabilityMode>('skills')
  const [mode, setMode] = embedded ? localTab : routeTab
  const gateway = useStoreSelector($gateway, g => (mode === 'connectors' ? g : null))

  const [query, setQuery] = useState('')
  const [capabilityView, setCapabilityView] = useState<CapabilityView>('installed')

  // Keep the docs iframe alive after the first Hub visit.
  const [hubMounted, setHubMounted] = useState(mode === 'hub')

  if (mode === 'hub' && !hubMounted) {
    setHubMounted(true)
  }

  const scope = useCapabilityScope({ fixedConnection, fixedProfile })

  // The two installed lists the tab pills count. They are fetched here, as a
  // pair, because the counts stay live for the tab the user is NOT on.
  const { data: skills, isError: skillsFailed, error: skillsError } = useSkillsQuery(scope.profile)
  const { data: toolsets, isError: toolsetsFailed } = useToolsetsQuery(scope.profile)
  const installedSkillNames = useMemo(() => new Set((skills ?? []).map(skill => skill.name)), [skills])

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

  // Rotating placeholder nudges from the user's own data — teach that search
  // understands categories and tool names, not just titles.
  const searchHints = useMemo(() => {
    if (mode === 'skills' && skills?.length) {
      return skillSearchTerms(skills).map(term => t.common.tryHint(term))
    }

    if (mode === 'toolsets' && toolsets?.length) {
      return toolsetSearchTerms(toolsets).map(term => t.common.tryHint(term))
    }

    return undefined
  }, [mode, skills, t, toolsets])

  // MCP and Plugins load independently of the installed Skills/Tools lists.
  const gated = mode === 'toolsets' || mode === 'skills'
  const pending = gated && !(skills && toolsets)

  const loadGate = !pending ? null : skillsFailed || toolsetsFailed ? (
    <PanelEmpty
      action={
        <Button onClick={() => void refreshCapabilities()} size="sm">
          {t.skills.refresh}
        </Button>
      }
      description={skillsError instanceof Error ? skillsError.message : undefined}
      icon="error"
      title={t.skills.skillsLoadFailed}
    />
  ) : (
    <PageLoader label={t.skills.loading} />
  )

  const tabContent = {
    hub: () => null,
    // The gateway instance backs ONLY the live `reload.mcp` RPC, and it is the
    // ACTIVE gateway's socket. A scope pinned elsewhere must not reload it.
    connectors: () => (
      <ConnectorsTab
        gateway={scope.crossBackend ? null : gateway}
        key={`connectors-${scope.key}`}
        profile={scope.profile}
      />
    ),
    // Agent plugins for the scoped profile (selector in the section header),
    // app-level desktop plugins, and the docs catalog picker underneath.
    plugins: () => (
      <PluginsTab
        key={`plugins-${scope.key}`}
        onQueryChange={setQuery}
        profile={scope.profile}
        query={query}
        scopeLabel={scope.label}
        scopeSelector={scope.options.length > 1 ? <CapabilityScopeSelector compact scope={scope} /> : undefined}
        view={capabilityView}
      />
    ),
    skills: () =>
      capabilityView === 'browse' ? (
        <SkillCatalog
          installedNames={installedSkillNames}
          key={scope.key}
          onQueryChange={setQuery}
          profile={scope.profile}
          query={query}
        />
      ) : (
        <SkillsTab
          key={`skills-${scope.key}`}
          onRefresh={() => void refreshCapabilities()}
          profile={scope.profile}
          query={query}
          skills={skills ?? []}
        />
      ),
    toolsets: () => (
      <ToolsetsTab key={`toolsets-${scope.key}`} profile={scope.profile} query={query} toolsets={toolsets ?? []} />
    )
  } satisfies Record<CapabilityMode, () => React.ReactNode>

  return (
    <PageSearchShell
      {...props}
      activeTab={mode}
      onSearchChange={setQuery}
      onTabChange={id => setMode(id as CapabilityMode)}
      // Connectors and Hub own their search fields.
      searchHidden={mode === 'connectors' || mode === 'hub'}
      searchHints={searchHints}
      searchPlaceholder={
        mode === 'plugins'
          ? t.catalog.searchPlugins
          : mode === 'skills'
            ? t.skills.searchSkills
            : t.skills.searchToolsets
      }
      searchValue={query}
      tabs={[
        { id: 'skills', label: t.skills.tabSkills, meta: skills?.length ?? null },
        { id: 'toolsets', label: t.skills.tabToolsets, meta: toolsets ? visibleToolsetCount(toolsets) : null },
        { id: 'connectors', label: t.connectorsPage.title },
        { id: 'plugins', label: t.skills.tabPlugins },
        { id: 'hub', label: t.skills.tabHub }
      ]}
    >
      <div className="flex h-full flex-col">
        {mode !== 'plugins' && <CapabilityScopeSelector scope={scope} />}
        {mode === 'skills' && (
          <CapabilityTabs
            actions={<UpdateSkillsButton profile={scope.profile} />}
            onChange={setCapabilityView}
            value={capabilityView}
          />
        )}
        <div className="flex min-h-0 flex-1 flex-col">
          <div className={mode === 'hub' ? 'hidden' : 'min-h-0 flex-1 overflow-hidden'}>
            {loadGate ?? tabContent[mode]()}
          </div>
          {hubMounted && (
            <EmbeddedHubPicker
              hidden={mode !== 'hub'}
              installedNames={installedSkillNames}
              profile={scope.profile}
              standalone
            />
          )}
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

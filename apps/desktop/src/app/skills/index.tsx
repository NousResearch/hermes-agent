import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { ArchiveSkillConfirmDialog } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { CountSkeleton } from '@/components/ui/skeleton'
import type { DesktopRosterAgent } from '@/global'
import {
  editLearningNode,
  getLearningNode,
  getProfiles,
  getSkillContent,
  getSkills,
  getToolsets,
  getUsageAnalytics,
  type ProfileScope,
  profileScopeKey,
  setSkillEnabled,
  setToolsetEnabled
} from '@/hermes'
import { useI18n } from '@/i18n'
import { isDesktopToolsetVisible } from '@/lib/desktop-toolsets'
import { compactNumber } from '@/lib/format'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { normalize } from '@/lib/text'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $gateway, activeGatewayConnectionId } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import {
  CapRow,
  DetailColumn,
  DetailPane,
  ListColumn,
  ListStrip,
  ListStripButton,
  ListStripMenu,
  type ListStripMenuToggle,
  MasterDetail,
  ToolChip
} from '../master-detail'
import { PanelEmpty, PanelPill } from '../overlays/panel'
import { PageSearchShell } from '../page-search-shell'
import { SETTINGS_ROUTE } from '../routes'
import { ComputerUsePanel } from '../settings/computer-use-panel'
import { asText, includesQuery, prettyName, toolNames, toolsetDisplayLabel } from '../settings/helpers'
import { TerminalBackendPanel } from '../settings/terminal-backend-panel'
import { ToolsetConfigPanel } from '../settings/toolset-config-panel'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { EmbeddedHubPicker } from './embedded-hub-picker'
import { McpTab } from './mcp-tab'
import { $skillsSortDesc, $toolsetsSortDesc } from './store'

// 'hub' is gone as a top-level tab — the Skills Hub browser lives inside the
// Skills tab now (EmbeddedHubPicker below the installed list). Legacy
// `?tab=hub` links fall back to 'skills' via useRouteEnumParam.
const SKILLS_MODES = ['skills', 'toolsets', 'mcp'] as const

// Skills + toolsets live in the RQ cache so switching tabs/pages paints the
// cached lists instantly (no reload flash) and mount only fires a deduped
// background refetch. A profile swap globally invalidates (see store/profile),
// so these plain keys refetch against the new backend automatically.
// Both are extended with the Capabilities scope key at the call sites so every
// scoped profile keeps its own cached copy (prefix invalidations still match).
const SKILLS_QUERY_KEY = ['skills-list'] as const
const TOOLSETS_QUERY_KEY = ['toolsets-list'] as const

// Per-tool call counts come from a 365-day message scan — heavy, and purely
// cosmetic (Toolsets usage badges). Cache the result module-wide with a TTL so
// bouncing between tabs/pages doesn't re-run the scan every time. Keyed by
// the Capabilities scope profile: analytics are profile-scoped, so a scope or
// app-profile switch must not show the previous profile's counts.
// `useRefreshHotkey` still forces a fresh pull.
const TOOL_CALLS_TTL_MS = 10 * 60 * 1000
const toolCallsCache = new Map<string, { at: number; value: Record<string, number> }>()

async function loadToolCalls(
  scopeKey: string,
  scopeProfile: ProfileScope,
  force = false
): Promise<Record<string, number>> {
  const cached = toolCallsCache.get(scopeKey)

  if (!force && cached && Date.now() - cached.at < TOOL_CALLS_TTL_MS) {
    return cached.value
  }

  const analytics = await getUsageAnalytics(365, scopeProfile)

  const value = Object.fromEntries((analytics.tools ?? []).map(e => [e.tool, e.count]))

  toolCallsCache.set(scopeKey, { at: Date.now(), value })

  return value
}

const usageOf = (skill: SkillInfo): number => (typeof skill.usage === 'number' ? skill.usage : 0)

const categoryFor = (skill: SkillInfo): string => asText(skill.category) || 'general'

// Row subtitle: category, with non-default origins badged.
function skillSubtitle(skill: SkillInfo): React.ReactNode {
  const category = prettyName(categoryFor(skill))
  const provenance = skill.provenance

  return (
    <>
      <span className="truncate">{category}</span>
      {provenance === 'agent' && (
        <Badge className="shrink-0 normal-case" variant="default">
          learned
        </Badge>
      )}
      {provenance === 'hub' && (
        <Badge className="shrink-0 normal-case" variant="muted">
          hub
        </Badge>
      )}
    </>
  )
}

function filteredSkills(skills: SkillInfo[], query: string, desc: boolean): SkillInfo[] {
  const q = normalize(query)
  const sign = desc ? 1 : -1

  return skills
    .filter(
      skill =>
        !q || includesQuery(skill.name, q) || includesQuery(skill.description, q) || includesQuery(skill.category, q)
    )
    .sort((a, b) => sign * (usageOf(b) - usageOf(a)) || asText(a.name).localeCompare(asText(b.name)))
}

const toolsetCalls = (toolset: ToolsetInfo, toolCalls: Record<string, number>): number =>
  toolNames(toolset).reduce((sum, name) => sum + (toolCalls[name] ?? 0), 0)

function filteredToolsets(
  toolsets: ToolsetInfo[],
  query: string,
  toolCalls: Record<string, number>,
  desc: boolean
): ToolsetInfo[] {
  const q = normalize(query)
  const sign = desc ? 1 : -1

  return toolsets
    .filter(toolset => {
      if (!isDesktopToolsetVisible(toolset.name)) {
        return false
      }

      if (!q) {
        return true
      }

      return (
        includesQuery(toolset.name, q) ||
        includesQuery(toolsetDisplayLabel(toolset), q) ||
        includesQuery(toolset.description, q) ||
        toolNames(toolset).some(name => includesQuery(name, q))
      )
    })
    .sort(
      (a, b) =>
        sign * (toolsetCalls(b, toolCalls) - toolsetCalls(a, toolCalls)) ||
        toolsetDisplayLabel(a).localeCompare(toolsetDisplayLabel(b))
    )
}

const visibleToolsetCount = (toolsets: ToolsetInfo[]) => toolsets.filter(ts => isDesktopToolsetVisible(ts.name)).length

interface SkillsViewProps extends React.ComponentProps<'section'> {
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

export function SkillsView({
  embedded = false,
  fixedConnection,
  fixedProfile,
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: SkillsViewProps) {
  const { t } = useI18n()
  // Both hooks run unconditionally (rules of hooks); embedded picks the local
  // one so tab clicks inside a dialog don't rewrite the page URL.
  const routeTab = useRouteEnumParam('tab', SKILLS_MODES, 'skills')
  const localTab = useState<(typeof SKILLS_MODES)[number]>('skills')
  const [mode, setMode] = embedded ? localTab : routeTab
  // $gateway only feeds the MCP tab — gate the subscription so Skills/Toolsets
  // tabs don't re-render on connect/disconnect/reconnect.
  const gateway = useStoreSelector($gateway, g => (mode === 'mcp' ? g : null))

  const [query, setQuery] = useState('')

  // The hub picker hosts a full docs-site iframe — the single most expensive
  // thing on this page. It mounts lazily (first time the Skills tab is shown)
  // and then STAYS mounted but hidden across tab switches, so bouncing to
  // Tools/MCP and back never reloads the site. Derived-state pattern: flips
  // once, during render, never back.
  const [hubMounted, setHubMounted] = useState(mode === 'skills')

  if (mode === 'skills' && !hubMounted) {
    setHubMounted(true)
  }

  // Capabilities scope selector: which profile's Skills/Tools/MCP config we're
  // editing — and on WHICH gateway. A profile belongs to one gateway, so on a
  // multi-connection desktop the selector offers every (connection, profile)
  // pair from the union agent roster and an explicit pick routes every
  // read/write to that machine's backend. Defaults to the app-wide active
  // profile (unchanged behavior). A `fixedProfile` (+ optional
  // `fixedConnection`) pins the scope outright (selector hidden).
  const activeProfile = useStore($activeGatewayProfile)
  const [scopeOverride, setScopeOverride] = useState<null | string | { connectionId: string; profile: string }>(null)

  const scopeProfile: ProfileScope = useMemo(
    () =>
      fixedProfile
        ? fixedConnection
          ? { connectionId: fixedConnection, profile: fixedProfile }
          : fixedProfile
        : (scopeOverride ?? activeProfile ?? null),
    [activeProfile, fixedConnection, fixedProfile, scopeOverride]
  )

  const scopeKey = profileScopeKey(scopeProfile)

  // The registry connection an explicit override pins — null for the ambient
  // active-profile path and for fixed-profile pins without a connection.
  const scopeConnectionId =
    scopeProfile && typeof scopeProfile === 'object' ? ((scopeProfile.connectionId ?? '').trim() || 'local') : null

  // Scoped to a DIFFERENT backend than the window's active gateway? The MCP
  // tab's live-reload RPC rides the active gateway socket, which would reload
  // the wrong machine — withhold it for cross-backend scopes.
  const crossBackendScope =
    scopeConnectionId !== null && scopeConnectionId !== (activeGatewayConnectionId() ?? 'local')

  const { data: profilesData } = useQuery({
    queryKey: ['capabilities-profiles'],
    queryFn: getProfiles,
    staleTime: 60_000,
    // Pinned scope never shows the selector, so don't fetch the roster for it.
    enabled: !fixedProfile
  })

  const profiles = profilesData?.profiles ?? []

  // v2 multi-connection registry: cheap local IPC to learn whether more than
  // one gateway is registered. Only then is the (heavier) union agent roster
  // fetched to feed the selector — single-connection setups keep the exact
  // legacy profiles list. Both feature-detected for older Electron mains.
  const registryBridge = window.hermesDesktop?.connections
  const rosterBridge = window.hermesDesktop?.getAgentRoster

  const { data: registryData } = useQuery({
    queryKey: ['capabilities-connections-registry'],
    queryFn: () => registryBridge!.list(),
    staleTime: 60_000,
    enabled: !fixedProfile && Boolean(registryBridge) && Boolean(rosterBridge)
  })

  const multiConnection = (registryData?.connections.length ?? 0) > 1

  const { data: rosterData } = useQuery({
    queryKey: ['capabilities-agent-roster'],
    queryFn: () => rosterBridge!(),
    staleTime: 60_000,
    enabled: !fixedProfile && multiConnection
  })

  const {
    data: skills,
    isError: skillsFailed,
    error: skillsError
  } = useQuery({
    queryKey: [...SKILLS_QUERY_KEY, scopeKey],
    queryFn: () => getSkills(scopeProfile),
    staleTime: 0
  })

  const { data: toolsets, isError: toolsetsFailed } = useQuery({
    queryKey: [...TOOLSETS_QUERY_KEY, scopeKey],
    queryFn: () => getToolsets(scopeProfile),
    staleTime: 0
  })

  // Optimistic write-through against the scoped Skills key: toggles/bulk/
  // archive repaint instantly; the next background refetch reconciles.
  const setSkills = useCallback(
    (fn: (cur: SkillInfo[] | undefined) => SkillInfo[] | undefined) =>
      queryClient.setQueryData<SkillInfo[]>([...SKILLS_QUERY_KEY, scopeKey], prev => fn(prev) ?? prev),
    [scopeKey]
  )

  // tool name -> call count over the analytics window. null = still loading
  // (badges show skeletons); {} = loaded empty / unavailable backend.
  const [toolCalls, setToolCalls] = useState<Record<string, number> | null>(null)
  // Bumped on profile switch so a slow analytics load from profile A can't set
  // toolCalls after the user moved to B.
  const toolCallsEpoch = useRef(0)
  const skillsSortDesc = useStore($skillsSortDesc)
  const toolsetsSortDesc = useStore($toolsetsSortDesc)
  const [bulkBusy, setBulkBusy] = useState(false)
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)
  const [selectedToolset, setSelectedToolset] = useState<string | null>(null)

  const refreshCapabilities = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: SKILLS_QUERY_KEY }),
      queryClient.invalidateQueries({ queryKey: TOOLSETS_QUERY_KEY })
    ])

    invalidateSlashCompletions()

    // An explicit refresh is the one time we bypass the analytics TTL — but
    // only if the badges are already on screen; otherwise let the lazy load
    // pick it up when Toolsets is first shown. Guard the async set against a
    // profile/scope switch landing before it resolves.
    if (toolCallsCache.size > 0) {
      const epoch = toolCallsEpoch.current

      loadToolCalls(scopeKey, scopeProfile, true)
        .then(value => toolCallsEpoch.current === epoch && setToolCalls(value))
        .catch(() => toolCallsEpoch.current === epoch && setToolCalls({}))
    }
  }, [scopeKey, scopeProfile])

  const refreshToolsets = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: TOOLSETS_QUERY_KEY })
  }, [])

  useRefreshHotkey(refreshCapabilities)

  // Per-tool call counts feed ONLY the Toolsets tab's usage badges/sort, and
  // the query behind them is a 365-day message scan — heavy. Fetch it lazily
  // the first time Toolsets is shown, never on Skills or MCP, so it can't
  // starve the MCP tab's config load. Absent → toolsets sort A–Z until it lands.
  useEffect(() => {
    if (mode !== 'toolsets' || toolCalls !== null) {
      return
    }

    let cancelled = false
    // Guard the setter by epoch too: when toolCalls is already null at switch
    // time, setToolCalls(null) is a no-op so this effect never re-runs to flip
    // `cancelled` — the epoch check catches that gap.
    const epoch = toolCallsEpoch.current
    const live = () => !cancelled && toolCallsEpoch.current === epoch

    loadToolCalls(scopeKey, scopeProfile)
      .then(value => live() && setToolCalls(value))
      .catch(() => live() && setToolCalls({}))

    return () => void (cancelled = true)
  }, [mode, scopeKey, scopeProfile, toolCalls])

  // On an app-wide profile switch the analytics cache is scope-keyed, but our
  // local toolCalls state isn't — leaving it non-null would keep the lazy
  // effect from ever re-running, so badges/sort would show the previous
  // profile's counts. Reset to null so the next Toolsets view reloads for the
  // active profile. The switch also drops any scope override — the user just
  // changed what "here" means, and a stale override pointing at the previous
  // selection would be surprising.
  useOnProfileSwitch(() => {
    toolCallsEpoch.current += 1
    setToolCalls(null)
    setScopeOverride(null)
  })

  const visibleSkills = useMemo(
    () => (skills ? filteredSkills(skills, query, skillsSortDesc) : []),
    [query, skills, skillsSortDesc]
  )

  const visibleToolsets = useMemo(
    () => (toolsets ? filteredToolsets(toolsets, query, toolCalls ?? {}, toolsetsSortDesc) : []),
    [query, toolCalls, toolsets, toolsetsSortDesc]
  )

  // Bulk actions ("All" master switch, "Disable unused") and the master-switch
  // state target the WHOLE tab, never the search-filtered view — a tab-wide
  // control that silently scoped to the current query would be a lie.
  const bulkSkills = skills ?? []
  const bulkToolsets = useMemo(() => (toolsets ?? []).filter(ts => isDesktopToolsetVisible(ts.name)), [toolsets])

  // Installed-name set for the hub picker's already-installed guard — the
  // UNFILTERED list on purpose (search must not make a skill look absent).
  const installedSkillNames = useMemo(() => new Set((skills ?? []).map(s => s.name)), [skills])

  // Rotating placeholder nudges from the user's own data — teach that search
  // understands categories and tool names, not just titles.
  const searchHints = useMemo(() => {
    if (mode === 'skills' && skills?.length) {
      const counts = new Map<string, number>()

      for (const skill of skills) {
        const key = categoryFor(skill)
        counts.set(key, (counts.get(key) || 0) + 1)
      }

      return [...counts.entries()]
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([category]) => t.common.tryHint(category.toLowerCase()))
    }

    if (mode === 'toolsets' && toolsets?.length) {
      return toolsets
        .filter(ts => isDesktopToolsetVisible(ts.name) && toolNames(ts).length > 0)
        .slice(0, 5)
        .map(ts => t.common.tryHint(toolNames(ts)[0]))
    }

    return undefined
  }, [mode, skills, toolsets, t])

  // Keep a valid selection: fall back to the first visible row when the
  // current selection is filtered out (or nothing is selected yet).
  const activeSkill = useMemo(
    () => visibleSkills.find(s => s.name === selectedSkill) ?? visibleSkills[0] ?? null,
    [selectedSkill, visibleSkills]
  )

  const activeToolset = useMemo(
    () => visibleToolsets.find(ts => ts.name === selectedToolset) ?? visibleToolsets[0] ?? null,
    [selectedToolset, visibleToolsets]
  )

  // Single toggles are optimistic and silent on success (the row repaints
  // immediately — a toast per flip would spam rapid customization). Errors
  // revert and notify.
  async function handleToggleSkill(skill: SkillInfo, enabled: boolean) {
    setSkills(current => current?.map(row => (row.name === skill.name ? { ...row, enabled } : row)) ?? current)

    try {
      await setSkillEnabled(skill.name, enabled, scopeProfile)
      // A disabled skill loses its `/name` command, so the composer's cached
      // `/` list has to be dropped along with the row repaint.
      invalidateSlashCompletions()
    } catch (err) {
      setSkills(
        current => current?.map(row => (row.name === skill.name ? { ...row, enabled: !enabled } : row)) ?? current
      )
      notifyError(err, t.skills.failedToUpdate(skill.name))
    }
  }

  async function handleToggleToolset(toolset: ToolsetInfo, enabled: boolean) {
    const scopedToolsetKey = [...TOOLSETS_QUERY_KEY, scopeKey]

    const writeScoped = (fn: (cur: ToolsetInfo[] | undefined) => ToolsetInfo[] | undefined) =>
      queryClient.setQueryData<ToolsetInfo[]>(scopedToolsetKey, prev => fn(prev) ?? prev)

    writeScoped(
      current =>
        current?.map(row => (row.name === toolset.name ? { ...row, enabled, available: enabled } : row)) ?? current
    )

    try {
      await setToolsetEnabled(toolset.name, enabled, scopeProfile)
    } catch (err) {
      writeScoped(
        current =>
          current?.map(row => (row.name === toolset.name ? { ...row, enabled: !enabled, available: !enabled } : row)) ??
          current
      )
      notifyError(err, t.skills.failedToUpdate(toolsetDisplayLabel(toolset)))
    }
  }

  // Sequential on purpose: each toggle is a config read-modify-write on the
  // backend; parallel calls would race the disabled-list save.
  async function bulkApply(skillTargets: SkillInfo[], toolsetTargets: ToolsetInfo[], enabled: boolean) {
    if (bulkBusy || skillTargets.length + toolsetTargets.length === 0) {
      return
    }

    setBulkBusy(true)

    let done = 0

    try {
      for (const row of skillTargets) {
        await setSkillEnabled(row.name, enabled, scopeProfile)
        setSkills(cur => cur?.map(r => (r.name === row.name ? { ...r, enabled } : r)) ?? cur)
        done += 1
      }

      for (const row of toolsetTargets) {
        await setToolsetEnabled(row.name, enabled, scopeProfile)
        queryClient.setQueryData<ToolsetInfo[]>(
          [...TOOLSETS_QUERY_KEY, scopeKey],
          cur => cur?.map(r => (r.name === row.name ? { ...r, enabled, available: enabled } : r)) ?? cur
        )
        done += 1
      }

      notify({ kind: 'success', title: t.skills.bulkUpdated(done), message: '' })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(mode === 'skills' ? t.skills.tabSkills : t.skills.tabToolsets))
    } finally {
      invalidateSlashCompletions()
      setBulkBusy(false)
    }
  }

  const bulkToggle = (enabled: boolean) =>
    mode === 'skills'
      ? bulkApply(
          bulkSkills.filter(row => row.enabled !== enabled),
          [],
          enabled
        )
      : bulkApply(
          [],
          bulkToolsets.filter(row => row.enabled !== enabled),
          enabled
        )

  // "Never used" = zero recorded activity. The pruning move for a 100+ skill
  // install: keep the workhorses, shed the noise.
  const disableUnused = () =>
    bulkApply(
      bulkSkills.filter(skill => skill.enabled && usageOf(skill) === 0),
      [],
      false
    )

  // One switch line covering enable-all/disable-all.
  const bulkSwitch = (allEnabled: boolean): ListStripMenuToggle => ({
    checked: allEnabled,
    disabled: bulkBusy,
    label: t.skills.all,
    onToggle: checked => void bulkToggle(checked)
  })

  const allSkillsEnabled = bulkSkills.length > 0 && bulkSkills.every(s => s.enabled)
  const allToolsetsEnabled = bulkToolsets.length > 0 && bulkToolsets.every(ts => ts.enabled)

  const sortButton = (desc: boolean, flip: () => void) => (
    <ListStripButton onClick={flip}>{desc ? t.skills.sortMostUsedDesc : t.skills.sortLeastUsedAsc}</ListStripButton>
  )

  // Full-bleed empty state, matching the MCP tab (spans both columns, not a
  // cramped note in the left rail). Query-aware, and says "tools" not the
  // internal "toolsets".
  const capabilityEmpty = (noun: string) => {
    const q = query.trim()

    return (
      <div className="flex h-full min-h-0 flex-1">
        <PanelEmpty
          description={q ? t.skills.emptyNothingMatches(q) : t.skills.emptyNoneAvailable(noun)}
          icon="search"
          title={t.skills.emptyNoneFound(noun)}
        />
      </div>
    )
  }

  // Learned/local skills are editable + archivable, mirroring the memory
  // graph (same /api/learning/node endpoints — delete archives, restorable
  // via `hermes curator restore`).
  const [skillEditor, setSkillEditor] = useState<null | { content: string; name: string }>(null)
  const [skillDraft, setSkillDraft] = useState('')
  const [skillSaving, setSkillSaving] = useState(false)
  const [archiveTarget, setArchiveTarget] = useState<null | string>(null)
  // Bumped on profile switch so an in-flight openSkillEditor fetch from profile
  // A can't reopen the editor with A's content after switching to B.
  const skillEditorEpoch = useRef(0)

  // A profile switch swaps the backend under the open editor/archive dialog —
  // their targets belong to profile A, so a save/archive would hit B. Drop them
  // so nothing edits or archives against the newly active profile.
  useOnProfileSwitch(() => {
    skillEditorEpoch.current += 1
    setSkillEditor(null)
    setSkillDraft('')
    setArchiveTarget(null)
  })

  const openSkillEditor = async (name: string) => {
    const epoch = skillEditorEpoch.current

    try {
      const node = await getLearningNode(name, scopeProfile)

      if (skillEditorEpoch.current !== epoch) {
        return
      }

      setSkillEditor({ content: node.content, name })
      setSkillDraft(node.content)
    } catch (err) {
      notifyError(err, name)
    }
  }

  const saveSkillEdit = async () => {
    if (!skillEditor) {
      return
    }

    setSkillSaving(true)

    try {
      await editLearningNode(skillEditor.name, skillDraft, scopeProfile)
      notify({
        kind: 'success',
        title: t.skills.skillUpdated,
        message: t.skills.appliesToNewSessions(skillEditor.name)
      })
      setSkillEditor(null)
      void refreshCapabilities()
    } catch (err) {
      notifyError(err, skillEditor.name)
    } finally {
      setSkillSaving(false)
    }
  }

  const skillEditorPane = skillEditor && (
    <DetailPane
      actions={
        <Button disabled={skillSaving} onClick={() => void saveSkillEdit()} size="xs">
          {skillSaving ? t.common.saving : t.common.save}
        </Button>
      }
      id="skill-editor"
      onClose={() => setSkillEditor(null)}
      title={<span className="text-[0.68rem] font-normal text-muted-foreground/60">{skillEditor.name}/SKILL.md</span>}
    >
      <CodeEditor
        filePath="SKILL.md"
        initialValue={skillEditor.content}
        key={skillEditor.name}
        onCancel={() => setSkillEditor(null)}
        onChange={setSkillDraft}
        onSave={() => void saveSkillEdit()}
      />
    </DetailPane>
  )

  // Selecting a different scope is the same staleness hazard as an app-wide
  // profile switch: in-flight analytics belong to the previous scope's cache
  // key, and an open skill editor / archive dialog targets the PREVIOUS
  // scope's skill (a save/archive would hit the new one). Reset both here —
  // this handler is the only way the scope changes besides an app profile
  // switch, which useOnProfileSwitch already covers.
  //
  // Selector option values are `connectionId::profile` on multi-connection
  // desktops (the roster path) and bare profile names otherwise, so one
  // handler decodes both.
  const changeScope = (value: string) => {
    const sep = value.indexOf('::')

    // Roster picks (`connectionId::profile`) stay objects — a `local::` pick
    // must PIN the local pool even while a remote gateway is active. Legacy
    // bare-name picks stay strings so cache keys and routing are unchanged.
    const next: ProfileScope =
      sep >= 0 ? { connectionId: value.slice(0, sep), profile: value.slice(sep + 2) } : value

    if (profileScopeKey(next) === scopeKey) {
      return
    }

    setScopeOverride(next as string | { connectionId: string; profile: string })
    toolCallsEpoch.current += 1
    setToolCalls(null)
    skillEditorEpoch.current += 1
    setSkillEditor(null)
    setSkillDraft('')
    setArchiveTarget(null)
  }

  // Scope-selector rows. Multi-connection desktops list every reachable
  // (connection, profile) agent from the union roster — the selected profile
  // is configured ON ITS OWN GATEWAY. Otherwise the legacy per-profile list.
  const scopeOptions: { key: string; label: string; value: string }[] = useMemo(() => {
    if (multiConnection && rosterData?.agents?.length) {
      const activeId = activeGatewayConnectionId() ?? 'local'

      return rosterData.agents.map((agent: DesktopRosterAgent) => ({
        key: `${agent.connectionId}::${agent.profile}`,
        label:
          agent.connectionId === activeId
            ? `${agent.profile} — ${agent.connectionLabel} (current)`
            : `${agent.profile} — ${agent.connectionLabel}`,
        value: `${agent.connectionId}::${agent.profile}`
      }))
    }

    return (profilesData?.profiles ?? []).map(p => ({
      key: p.name,
      label: p.is_default ? 'Hermes (default)' : p.name,
      value: p.name
    }))
  }, [multiConnection, profilesData, rosterData])

  // The selector's current value must match one option's value exactly. On the
  // roster path an ambient (non-override) scope is the active gateway's
  // profile, which lives at `activeConnectionId::profile`.
  const scopeSelectValue = useMemo(() => {
    if (scopeProfile && typeof scopeProfile === 'object') {
      return `${(scopeProfile.connectionId ?? '').trim() || 'local'}::${scopeProfile.profile ?? ''}`
    }

    if (multiConnection && rosterData?.agents?.length) {
      return `${activeGatewayConnectionId() ?? 'local'}::${normalizeProfileKey(scopeProfile)}`
    }

    return scopeProfile ?? ''
  }, [multiConnection, rosterData, scopeProfile])

  // Scope selector, shown above EVERY Capabilities tab (Skills, Tools, MCP,
  // Browse Hub). Lets the user configure ANY profile's capabilities — on any
  // registered gateway — without switching the whole app. Only meaningful
  // with >1 option; hidden otherwise to avoid clutter.
  const profileScopeSelector =
    scopeOptions.length > 1 ? (
      <div className="flex items-center gap-2 border-b border-(--ui-stroke-secondary) px-3 py-2">
        <span className="text-[0.7rem] font-medium text-(--ui-text-tertiary)">{t.skills.configuringProfile}</span>
        <Select onValueChange={changeScope} value={scopeSelectValue}>
          <SelectTrigger className="h-7 w-56 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {scopeOptions.map(option => (
              <SelectItem key={option.key} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    ) : null

  return (
    <PageSearchShell
      {...props}
      activeTab={mode}
      onSearchChange={setQuery}
      onTabChange={id => setMode(id as (typeof SKILLS_MODES)[number])}
      // MCP manages a handful of entries with the editor right there —
      // searching it is noise.
      searchHidden={mode === 'mcp'}
      searchHints={searchHints}
      searchPlaceholder={mode === 'skills' ? t.skills.searchSkills : t.skills.searchToolsets}
      searchValue={query}
      tabs={[
        { id: 'skills', label: t.skills.tabSkills, meta: skills?.length ?? null },
        { id: 'toolsets', label: t.skills.tabToolsets, meta: toolsets ? visibleToolsetCount(toolsets) : null },
        { id: 'mcp', label: t.skills.tabMcp }
      ]}
    >
      {/* One shared column: the scope selector sits above whichever tab is
          active, so Skills / Tools / MCP all read and write the SAME selected
          profile. */}
      <div className="flex h-full flex-col">
        {profileScopeSelector}
        <div className="flex min-h-0 flex-1 flex-col">
          <div className={mode === 'skills' ? 'min-h-40 flex-1 overflow-hidden' : 'min-h-0 flex-1'}>
            {mode === 'mcp' ? (
              // The gateway instance backs ONLY the live `reload.mcp` RPC, and
              // it is the ACTIVE gateway's socket — for a scope pinned to a
              // different backend that RPC would hot-reload the wrong
              // machine's MCP servers, so it is withheld (config edits still
              // apply on that backend's next session).
              <McpTab gateway={crossBackendScope ? null : gateway} key={`mcp-${scopeKey}`} profile={scopeProfile} />
            ) : (skillsFailed || toolsetsFailed) && (!skills || !toolsets) ? (
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
            ) : !skills || !toolsets ? (
              <PageLoader label={t.skills.loading} />
            ) : mode === 'skills' ? (
              // Installed skills on top, the Skills Hub browser underneath —
              // discovery sits with management. The list region keeps a floor
              // (min-h-40, on the wrapper above) so a tall hub viewport or a
              // short window shrinks the HUB, never the list: the sort strip
              // and "changes apply" footer can no longer be starved to 0px
              // and painted over by the hub header.
              visibleSkills.length === 0 ? (
                capabilityEmpty('skills')
              ) : (
                <MasterDetail pane={skillEditorPane} resizeId="capabilities-split" split="wide">
                  <ListColumn
                    header={
                      <ListStrip
                        left={sortButton(skillsSortDesc, () => $skillsSortDesc.set(!$skillsSortDesc.get()))}
                        right={
                          <ListStripMenu
                            items={[
                              {
                                disabled: bulkBusy,
                                label: t.skills.disableUnused,
                                onSelect: () => void disableUnused()
                              }
                            ]}
                            label={t.skills.tabSkills}
                            toggle={bulkSwitch(allSkillsEnabled)}
                          />
                        }
                      />
                    }
                  >
                    {visibleSkills.map(skill => (
                      <CapRow
                        active={activeSkill?.name === skill.name}
                        busy={bulkBusy}
                        enabled={skill.enabled}
                        key={skill.name}
                        meta={usageOf(skill) > 0 ? `×${compactNumber(usageOf(skill))}` : undefined}
                        onSelect={() => setSelectedSkill(skill.name)}
                        onToggle={enabled => void handleToggleSkill(skill, enabled)}
                        subtitle={skillSubtitle(skill)}
                        title={skill.name}
                        toggleLabel={skill.name}
                      />
                    ))}
                  </ListColumn>
                  <DetailColumn footer={t.skills.changesApplyNewSessions}>
                    {activeSkill && (
                      <SkillDetail
                        onArchive={() => setArchiveTarget(activeSkill.name)}
                        onEdit={() => void openSkillEditor(activeSkill.name)}
                        profile={scopeProfile}
                        skill={activeSkill}
                      />
                    )}
                  </DetailColumn>
                </MasterDetail>
              )
            ) : visibleToolsets.length === 0 ? (
              capabilityEmpty('tools')
            ) : (
              <MasterDetail resizeId="capabilities-split" split="wide">
                <ListColumn
                  header={
                    <ListStrip
                      left={sortButton(toolsetsSortDesc, () => $toolsetsSortDesc.set(!$toolsetsSortDesc.get()))}
                      right={<ListStripMenu label={t.skills.tabToolsets} toggle={bulkSwitch(allToolsetsEnabled)} />}
                    />
                  }
                >
                  {visibleToolsets.map(toolset => {
                    const label = toolsetDisplayLabel(toolset)
                    const calls = toolCalls ? toolsetCalls(toolset, toolCalls) : null

                    return (
                      <CapRow
                        active={activeToolset?.name === toolset.name}
                        busy={bulkBusy}
                        enabled={toolset.enabled}
                        key={toolset.name}
                        meta={
                          calls === null ? (
                            <CountSkeleton />
                          ) : calls > 0 ? (
                            `×${compactNumber(calls)}`
                          ) : (
                            `${toolNames(toolset).length} tools`
                          )
                        }
                        onSelect={() => setSelectedToolset(toolset.name)}
                        onToggle={checked => void handleToggleToolset(toolset, checked)}
                        subtitle={asText(toolset.description)}
                        title={label}
                        toggleLabel={t.skills.toggleToolset(label, !toolset.enabled)}
                      />
                    )
                  })}
                </ListColumn>
                <DetailColumn footer={t.skills.changesApplyNewSessions}>
                  {activeToolset && (
                    <ToolsetDetail
                      onConfiguredChange={refreshToolsets}
                      profile={scopeProfile}
                      toolCalls={toolCalls ?? {}}
                      toolset={activeToolset}
                    />
                  )}
                </DetailColumn>
              </MasterDetail>
            )}
          </div>
          {/* Hub picker OUTSIDE the tab ternary: it lazy-mounts the first time
              Skills is shown, then stays mounted (hidden) across Tools/MCP so
              the docs-site iframe never reloads on a tab bounce. No scope key
              on purpose — the picker fetches nothing; scope rides the
              `profile` prop into each install call, and remounting on scope
              change would reload the whole site for no data benefit. */}
          {hubMounted && (
            <EmbeddedHubPicker hidden={mode !== 'skills'} installedNames={installedSkillNames} profile={scopeProfile} />
          )}
        </div>
      </div>
      {archiveTarget && (
        <ArchiveSkillConfirmDialog
          onApply={() => {
            const name = archiveTarget
            const snapshot = skills

            setSkills(current => current?.filter(skill => skill.name !== name) ?? current)
            invalidateSlashCompletions()

            if (skillEditor?.name === name) {
              setSkillEditor(null)
            }

            return () => setSkills(() => snapshot)
          }}
          onClose={() => setArchiveTarget(null)}
          onFailure={(err, name) => notifyError(err, name)}
          open
          profile={scopeProfile}
          skillId={archiveTarget}
          skillName={archiveTarget}
        />
      )}
    </PageSearchShell>
  )
}

// Feature-detection flag for plugins (Bot Mode): TRUE means this build's
// SkillsView routes `fixedConnection` to the pinned connection's backend.
// Older builds export SkillsView WITHOUT the prop — passing it there would
// silently read/write the ACTIVE gateway under the remote bot's profile name,
// which is exactly the wrong-machine bug the prop exists to prevent. A static
// property is probe-able without rendering.
SkillsView.supportsFixedConnection = true as const

// Shared inspector header — mirrors Messaging's PlatformDetail so Skills and
// Tools share one title/description block and tab switches don't jump.
function DetailHeader({
  description,
  pills,
  title
}: {
  description: React.ReactNode
  pills?: React.ReactNode
  title: string
}) {
  return (
    <header>
      <div className="flex min-h-6 flex-wrap items-center gap-2">
        <h3 className="min-w-0 truncate text-[0.9375rem] font-semibold tracking-tight">{title}</h3>
        {pills}
      </div>
      <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </header>
  )
}

// Frontmatter parse for display: the YAML block between the leading `---`
// fences, flattened to top-level `key: value` rows (nested blocks render as
// their raw indented text). Display-only — never fed back to the backend.
function parseFrontmatter(content: string): { body: string; meta: [string, string][] } {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(content)

  if (!match) {
    return { body: content, meta: [] }
  }

  const meta: [string, string][] = []
  let currentKey: null | string = null
  let block: string[] = []

  const flush = () => {
    if (currentKey !== null) {
      meta.push([currentKey, block.join('\n').trim()])
    }

    currentKey = null
    block = []
  }

  for (const line of match[1].split(/\r?\n/)) {
    const kv = /^(\w[\w-]*):\s?(.*)$/.exec(line)

    if (kv) {
      flush()
      currentKey = kv[1]
      block = kv[2] ? [kv[2]] : []
    } else if (currentKey !== null) {
      block.push(line.replace(/^ {2}/, ''))
    }
  }

  flush()

  return { body: content.slice(match[0].length), meta }
}

function SkillDetail({
  onArchive,
  onEdit,
  profile,
  skill
}: {
  onArchive: () => void
  onEdit: () => void
  profile?: ProfileScope
  skill: SkillInfo
}) {
  const { t } = useI18n()
  // Only learned/local skills are the user's to rewrite or archive — bundled
  // and hub skills are managed by their sources.
  const editable = skill.provenance === 'agent'

  // The FULL skill — frontmatter metadata + complete SKILL.md body — for any
  // provenance, scoped to the Capabilities profile selector. The row list only
  // carries name/description; the pane shows the whole thing.
  const contentQuery = useQuery({
    queryKey: ['skill-content', skill.name, profileScopeKey(profile)],
    queryFn: () => getSkillContent(skill.name, profile),
    staleTime: 60_000
  })

  const parsed = useMemo(
    () => (contentQuery.data ? parseFrontmatter(contentQuery.data.content) : null),
    [contentQuery.data]
  )

  return (
    <>
      <DetailHeader
        description={asText(skill.description) || t.skills.noDescription}
        pills={
          <>
            <PanelPill>{prettyName(categoryFor(skill))}</PanelPill>
            {skill.provenance && skill.provenance !== 'bundled' && (
              <PanelPill tone={skill.provenance === 'agent' ? 'good' : 'muted'}>
                {t.skills.provenance[skill.provenance]}
              </PanelPill>
            )}
          </>
        }
        title={skill.name}
      />
      {editable && (
        <div className="flex items-center gap-2">
          <Button onClick={onEdit} size="xs" variant="text">
            {t.skills.edit}
          </Button>
          <Button className="text-destructive hover:text-destructive" onClick={onArchive} size="xs" variant="text">
            {t.skills.archive}
          </Button>
        </div>
      )}
      {parsed && parsed.meta.length > 0 && (
        <div className="grid gap-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3">
          {parsed.meta.map(([key, value]) => (
            <div className="flex gap-2 text-[0.68rem] leading-4" key={key}>
              <span className="w-24 shrink-0 font-medium text-(--ui-text-tertiary)">{key}</span>
              <span className="min-w-0 whitespace-pre-wrap break-words text-(--ui-text-secondary)">{value}</span>
            </div>
          ))}
        </div>
      )}
      {contentQuery.isLoading ? (
        <CountSkeleton />
      ) : parsed ? (
        <pre
          className="overflow-auto whitespace-pre-wrap wrap-break-word rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) p-3 font-mono text-[0.68rem] leading-relaxed"
          data-selectable-text="true"
        >
          {parsed.body.trim() || t.skills.noDescription}
        </pre>
      ) : null}
    </>
  )
}

function ToolsetDetail({
  toolset,
  toolCalls,
  onConfiguredChange,
  profile
}: {
  toolset: ToolsetInfo
  toolCalls: Record<string, number>
  onConfiguredChange: () => void
  profile?: ProfileScope
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const tools = toolNames(toolset)
  const label = toolsetDisplayLabel(toolset)

  return (
    <>
      {/* "Configured" as a resting state is noise — only the warn state earns a pill. */}
      <DetailHeader
        description={asText(toolset.description) || t.skills.noDescription}
        pills={!toolset.configured && <PanelPill tone="warn">{t.skills.needsKeys}</PanelPill>}
        title={label}
      />
      {tools.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tools.map(name => (
            <ToolChip key={name}>
              {name}
              {(toolCalls[name] ?? 0) > 0 && (
                <span className="ml-1 text-(--ui-text-quaternary)">×{compactNumber(toolCalls[name])}</span>
              )}
            </ToolChip>
          ))}
        </div>
      )}
      {toolset.name === 'vision' && (
        // Vision has no provider matrix — model resolution runs through the
        // auxiliary model config. Point at the actual home (Settings → Models,
        // aux "vision" row) via an internal deep link instead of leaving the
        // detail pane empty.
        <div className="grid gap-1.5">
          <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {t.skills.visionModelHint}
          </p>
          <div>
            <Button
              onClick={() => navigate(`${SETTINGS_ROUTE}?tab=config:model&aux=vision`)}
              size="xs"
              variant="textStrong"
            >
              {t.skills.visionModelLink}
            </Button>
          </div>
        </div>
      )}
      {toolset.name === 'computer_use' && <ComputerUsePanel onConfiguredChange={onConfiguredChange} />}
      {toolset.name === 'terminal' && <TerminalBackendPanel onConfiguredChange={onConfiguredChange} />}
      <ToolsetConfigPanel
        key={`${toolset.name}:${profileScopeKey(profile)}`}
        onConfiguredChange={onConfiguredChange}
        profile={profile}
        toolset={toolset.name}
      />
    </>
  )
}

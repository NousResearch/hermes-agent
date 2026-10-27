import { compactNumber } from '@hermes/shared'
import { useCallback, useMemo, useState } from 'react'

import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import {
  setToolsetEnabled,
  setToolsetGroupEnabled,
  type ProfileScope
} from '@/hermes'
import { useI18n } from '@/i18n'
import { isDesktopToolsetVisible } from '@/lib/desktop-toolsets'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { notify, notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import { CatalogSwitch } from '../catalog/catalog-switch'
import { CapabilityEmpty } from '../primitives'
import { CapabilitySearch } from '../ui/capability-search'
import { CatalogSurface, type CatalogSurfaceItem, type CatalogSurfaceSection } from '../ui/catalog-surface'

import { useToolCalls } from './tool-calls'
import { ToolsetDetail } from './toolset-detail'
import {
  filteredToolsets,
  TOOLSET_ESSENTIALS,
  TOOLSET_FEATURE_COUNT,
  TOOLSET_FEATURE_MIN,
  toolsetCalls,
  TOOLSETS_QUERY_KEY,
  toolsetsQueryKey
} from './toolsets-data'

// Display order for the functional-group sections; unknown groups trail.
const GROUP_ORDER = ['web', 'system', 'media', 'knowledge', 'integrations', 'plugins', 'other']
const groupRank = (group: string) => {
  const i = GROUP_ORDER.indexOf(group)
  return i === -1 ? GROUP_ORDER.length : i
}

interface ToolsetsTabProps {
  /** The scope's toolset list, straight from the shell's query. */
  toolsets: ToolsetInfo[]
  /** The (connection, profile) scope every read and write routes to. */
  profile: ProfileScope
  query: string
  onQueryChange: (query: string) => void
  /** Data-derived "Try …" placeholder nudges. */
  hints?: string[]
}

/** THE Tools tab: toolsets as a catalog (feature row, themed lists, developer
 *  banner) with each toolset's settings in a dialog. */
export function ToolsetsTab({ hints, onQueryChange, profile, query, toolsets }: ToolsetsTabProps) {
  const { t } = useI18n()
  const ts = t.catalog.toolsets
  const toolCalls = useToolCalls(profile)
  const [bulkBusy, setBulkBusy] = useState(false)
  const [selectedToolset, setSelectedToolset] = useState<string | null>(null)

  // Optimistic write-through against the scoped Tools key: toggles repaint
  // instantly; the next background refetch reconciles.
  const setToolsets = useCallback(
    (fn: (cur: ToolsetInfo[] | undefined) => ToolsetInfo[] | undefined) =>
      queryClient.setQueryData<ToolsetInfo[]>(toolsetsQueryKey(profile), prev => fn(prev) ?? prev),
    [profile]
  )

  const refreshToolsets = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: TOOLSETS_QUERY_KEY })
  }, [])

  // Most-used first; absent counts sort A–Z until the analytics scan lands.
  const visibleToolsets = useMemo(
    () => filteredToolsets(toolsets, query, toolCalls ?? {}, true),
    [query, toolCalls, toolsets]
  )

  // The list renders as TOOLSET_GROUPS sections (header + its rows, list sort
  // preserved inside a section). Switch state and the header switch's write
  // target the WHOLE group, never the search-filtered view — same contract as
  // the strip-menu bulk items.
  const groupedToolsets = useMemo(() => {
    const byGroup = new Map<string, ToolsetInfo[]>()
    for (const ts of visibleToolsets) {
      const key = ts.group ?? 'other'
      byGroup.set(key, [...(byGroup.get(key) ?? []), ts])
    }

    return Array.from(byGroup.entries()).sort((a, b) => groupRank(a[0]) - groupRank(b[0]))
  }, [visibleToolsets])

  // Bulk actions and the master-switch state target the WHOLE tab, never the
  // search-filtered view — a tab-wide control that silently scoped to the
  // current query would be a lie.
  const bulkToolsets = useMemo(() => toolsets.filter(ts => isDesktopToolsetVisible(ts.name)), [toolsets])

  // Whole-group truth for the section-header switches: size + is any child on.
  // The header is a kill switch — mixed reads as ON; clicking it while on
  // disables the whole group.
  const groupState = useMemo(() => {
    const state = new Map<string, { anyEnabled: boolean; size: number }>()
    const rowsByGroup = new Map<string, ToolsetInfo[]>()
    for (const ts of bulkToolsets) {
      const key = ts.group ?? 'other'
      rowsByGroup.set(key, [...(rowsByGroup.get(key) ?? []), ts])
    }

    for (const [group, rows] of rowsByGroup) {
      state.set(group, { anyEnabled: rows.some(ts => ts.enabled), size: rows.length })
    }

    return state
  }, [bulkToolsets])

  // Keep a valid selection: fall back to the first visible row when the
  // current selection is filtered out (or nothing is selected yet).
  const activeToolset = useMemo(
    () => visibleToolsets.find(ts => ts.name === selectedToolset) ?? null,
    [selectedToolset, visibleToolsets]
  )

  // Single toggles are optimistic and silent on success (the row repaints
  // immediately — a toast per flip would spam rapid customization). Errors
  // revert and notify.
  async function handleToggleToolset(toolset: ToolsetInfo, enabled: boolean) {
    setToolsets(
      current =>
        current?.map(row => (row.name === toolset.name ? { ...row, enabled, available: enabled } : row)) ?? current
    )

    try {
      await setToolsetEnabled(toolset.name, enabled, profile)
    } catch (err) {
      setToolsets(
        current =>
          current?.map(row => (row.name === toolset.name ? { ...row, enabled: !enabled, available: !enabled } : row)) ??
          current
      )
      notifyError(err, t.skills.failedToUpdate(toolsetDisplayLabel(toolset)))
    }
  }

  // Sequential on purpose: each toggle is a config read-modify-write on the
  // backend; parallel calls would race the disabled-list save.
  async function bulkApply(targets: ToolsetInfo[], enabled: boolean) {
    if (bulkBusy || targets.length === 0) {
      return
    }

    setBulkBusy(true)

    let done = 0

    try {
      for (const row of targets) {
        await setToolsetEnabled(row.name, enabled, profile)
        setToolsets(cur => cur?.map(r => (r.name === row.name ? { ...r, enabled, available: enabled } : r)) ?? cur)
        done += 1
      }

      notify({ kind: 'success', title: t.skills.bulkUpdated(done), message: '' })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(t.skills.tabToolsets))
    } finally {
      invalidateSlashCompletions()
      setBulkBusy(false)
    }
  }

  // Same search row as every other Capabilities tab; it stays put when a query
  // empties the list.
  const search = (
    <header className="catalog-search-header shrink-0">
      <CapabilitySearch
        hints={hints}
        onChange={onQueryChange}
        placeholder={t.skills.searchToolsets}
        scope={t.skills.tabToolsets}
        value={query}
      />
    </header>
  )

  // Per-group bulk: one backend request per display group (TOOLSET_GROUPS);
  // rows repaint from response names.
  async function bulkApplyGroup(group: string, enabled: boolean) {
    if (bulkBusy) {
      return
    }

    setBulkBusy(true)

    try {
      const result = await setToolsetGroupEnabled(
        group,
        enabled,
        profile,
        bulkToolsets.filter(ts => (ts.group ?? 'other') === group).map(ts => ts.name)
      )
      const names = new Set(result.names)
      setToolsets(cur => cur?.map(r => (names.has(r.name) ? { ...r, enabled, available: enabled } : r)) ?? cur)
      notify({ kind: 'success', title: t.skills.bulkUpdated(result.names.length), message: '' })
    } catch (err) {
      notifyError(err, t.skills.failedToUpdate(t.skills.toolsetGroupName(group)))
    } finally {
      invalidateSlashCompletions()
      setBulkBusy(false)
    }
  }

  if (visibleToolsets.length === 0) {
    return <div className="flex h-full min-h-0 flex-col">{search}<CapabilityEmpty noun="tools" query={query} /></div>
  }

  const item = (toolset: ToolsetInfo, layout: CatalogSurfaceSection['layout']): CatalogSurfaceItem => {
    const calls = toolCalls ? toolsetCalls(toolset, toolCalls) : 0
    const count = toolNames(toolset).length
    const tools = ts.toolCount(count)
    const label = toolsetDisplayLabel(toolset)

    return {
      id: toolset.name,
      name: label,
      description: asText(toolset.description),
      eyebrow: count ? tools : undefined,
      badge: calls > 0 ? ts.callCount(compactNumber(calls), calls) : layout === 'list' && count ? tools : undefined,
      action: (
        <CatalogSwitch
          aria-label={t.skills.toggleToolset(label, !toolset.enabled)}
          checked={toolset.enabled}
          disabled={bulkBusy}
          onCheckedChange={checked => void handleToggleToolset(toolset, checked)}
        />
      ),
      onOpen: () => setSelectedToolset(toolset.name)
    }
  }

  // The feature row is the four most-used toolsets once there is real usage to
  // rank by; before that it's the essentials. Everything else sits under what it does.
  const used = visibleToolsets.slice(0, TOOLSET_FEATURE_COUNT)
  const ranked = !!toolCalls && used.length === TOOLSET_FEATURE_COUNT && used.every(toolset => toolsetCalls(toolset, toolCalls) > 0)
  const featured = query || visibleToolsets.length <= TOOLSET_FEATURE_MIN ? [] : ranked ? used : visibleToolsets.filter(toolset => TOOLSET_ESSENTIALS.includes(toolset.name))
  const sections: CatalogSurfaceSection[] = [
    ...(featured.length
      ? [
          {
            id: 'featured',
            label: ranked ? ts.mostUsed : ts.essentials,
            blurb: ranked ? ts.mostUsedBlurb : ts.essentialsBlurb,
            layout: 'feature' as const,
            items: featured.map(toolset => item(toolset, 'feature'))
          }
        ]
      : []),
    // Functional groups (the backend's `group`) with a bulk switch on each
    // header — the config surface this tab exists for. Featured rows are not
    // repeated inside their group; the header switch targets the WHOLE group
    // (groupState), never the search-filtered rows.
    ...groupedToolsets.map(([group, rows]) => {
      const state = groupState.get(group)

      return {
        id: `group-${group}`,
        label: t.skills.toolsetGroupName(group),
        layout: 'list' as const,
        action: (
          <CatalogSwitch
            aria-label={t.skills.groupEnableAll(t.skills.toolsetGroupName(group), state?.size ?? rows.length)}
            checked={state?.anyEnabled ?? false}
            disabled={bulkBusy}
            onCheckedChange={checked => void bulkApplyGroup(group, checked)}
          />
        ),
        items: rows.filter(toolset => !featured.includes(toolset)).map(toolset => item(toolset, 'list'))
      }
    })
  ]

  return (
    <div className="flex h-full min-h-0 flex-col">
      {search}
      <CatalogSurface
        actions={
          <CatalogSwitch
            aria-label={t.skills.all}
            checked={bulkToolsets.length > 0 && bulkToolsets.every(ts => ts.enabled)}
            disabled={bulkBusy}
            onCheckedChange={checked => void bulkApply(bulkToolsets.filter(row => row.enabled !== checked), checked)}
          />
        }
        kind="plugins"
        sections={sections}
      />
      <Dialog onOpenChange={open => { if (!open) {setSelectedToolset(null)} }} open={selectedToolset !== null}>
        <DialogContent className="max-w-2xl">
          <DialogTitle className="sr-only">{activeToolset ? toolsetDisplayLabel(activeToolset) : t.skills.tabToolsets}</DialogTitle>
          {activeToolset && (
            <ToolsetDetail onConfiguredChange={refreshToolsets} profile={profile} toolCalls={toolCalls ?? {}} toolset={activeToolset} />
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useState } from 'react'

import { CountSkeleton } from '@/components/ui/skeleton'
import {
  setToolsetEnabled,
  setToolsetGroupEnabled,
  type ProfileScope
} from '@/hermes'
import { useI18n } from '@/i18n'
import { isDesktopToolsetVisible } from '@/lib/desktop-toolsets'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { notify, notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import {
  CapRow,
  DetailColumn,
  ListColumn,
  ListStrip,
  ListStripMenu,
  type ListStripMenuToggle,
  MasterDetail
} from '../../master-detail'
import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import { CapabilityEmpty, GroupHeaderRow, SortButton } from '../primitives'

import { useToolCalls } from './tool-calls'
import { ToolsetDetail } from './toolset-detail'
import { filteredToolsets, toolsetCalls, TOOLSETS_QUERY_KEY, toolsetsQueryKey } from './toolsets-data'

// Display order for TOOLSET_GROUPS sections; unknown groups trail.
const GROUP_ORDER = ['web', 'system', 'media', 'knowledge', 'integrations', 'plugins', 'other']
const groupRank = (group: string) => {
  const i = GROUP_ORDER.indexOf(group)
  return i === -1 ? GROUP_ORDER.length : i
}

// Sort direction for the Tools list — persisted so the tab remembers
// most/least-used across navigations and restarts.
const $toolsetsSortDesc = persistentAtom('hermes.desktop.capabilities.toolsetsSortDesc', true, Codecs.bool)

interface ToolsetsTabProps {
  /** The scope's toolset list, straight from the shell's query. */
  toolsets: ToolsetInfo[]
  /** The (connection, profile) scope every read and write routes to. */
  profile: ProfileScope
  query: string
}

/** THE Tools tab: the toolset list with its usage badges, and the inspector
 *  that mounts each toolset's settings panels. */
export function ToolsetsTab({ profile, query, toolsets }: ToolsetsTabProps) {
  const { t } = useI18n()
  const toolsetsSortDesc = useStore($toolsetsSortDesc)
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

  // Absent counts sort the list A–Z until the analytics scan lands.
  const visibleToolsets = useMemo(
    () => filteredToolsets(toolsets, query, toolCalls ?? {}, toolsetsSortDesc),
    [query, toolCalls, toolsets, toolsetsSortDesc]
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
    () => visibleToolsets.find(ts => ts.name === selectedToolset) ?? visibleToolsets[0] ?? null,
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

  // One switch line covering enable-all/disable-all.
  const bulkSwitch: ListStripMenuToggle = {
    checked: bulkToolsets.length > 0 && bulkToolsets.every(ts => ts.enabled),
    disabled: bulkBusy,
    label: t.skills.all,
    onToggle: checked =>
      void bulkApply(
        bulkToolsets.filter(row => row.enabled !== checked),
        checked
      )
  }

  // Per-group bulk: one backend request per display group (TOOLSET_GROUPS);
  // rows repaint from response names.
  async function bulkApplyGroup(group: string, enabled: boolean) {
    if (bulkBusy) {
      return
    }

    setBulkBusy(true)

    try {
      const result = await setToolsetGroupEnabled(group, enabled, profile)
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

  const groupBulkItems = useMemo(() => {
    const byGroup = new Map<string, ToolsetInfo[]>()
    for (const ts of bulkToolsets) {
      const key = ts.group ?? 'other'
      byGroup.set(key, [...(byGroup.get(key) ?? []), ts])
    }

    return Array.from(byGroup.entries()).map(([group, rows]) => {
      const anyEnabled = rows.some(ts => ts.enabled)
      return {
        disabled: bulkBusy,
        label: anyEnabled
          ? t.skills.groupDisableAll(t.skills.toolsetGroupName(group), rows.length)
          : t.skills.groupEnableAll(t.skills.toolsetGroupName(group), rows.length),
        onSelect: () => void bulkApplyGroup(group, !anyEnabled)
      }
    })
  }, [bulkBusy, bulkToolsets, t])

  if (visibleToolsets.length === 0) {
    return <CapabilityEmpty noun="tools" query={query} />
  }

  return (
    <MasterDetail resizeId="capabilities-split" split="wide">
      <ListColumn
        header={
          <ListStrip
            left={<SortButton desc={toolsetsSortDesc} onFlip={() => $toolsetsSortDesc.set(!$toolsetsSortDesc.get())} />}
            right={<ListStripMenu items={groupBulkItems} label={t.skills.tabToolsets} toggle={bulkSwitch} />}
          />
        }
      >
        {groupedToolsets.map(([group, rows]) => {
          const state = groupState.get(group)
          const size = state?.size ?? rows.length

          return (
            <div className="contents" key={group}>
              <GroupHeaderRow
                busy={bulkBusy}
                count={size}
                enabled={state?.anyEnabled ?? false}
                label={t.skills.toolsetGroupName(group)}
                onToggle={checked => void bulkApplyGroup(group, checked)}
              />
              {rows.map(toolset => {
                const label = toolsetDisplayLabel(toolset)
                const calls = toolCalls ? toolsetCalls(toolset, toolCalls) : null

                return (
                  <CapRow
                    active={activeToolset?.name === toolset.name}
                    busy={bulkBusy}
                    enabled={toolset.enabled}
                    indent
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
            </div>
          )
        })}
      </ListColumn>
      <DetailColumn footer={t.skills.changesApplyNewSessions}>
        {activeToolset && (
          <ToolsetDetail
            onConfiguredChange={refreshToolsets}
            profile={profile}
            toolCalls={toolCalls ?? {}}
            toolset={activeToolset}
          />
        )}
      </DetailColumn>
    </MasterDetail>
  )
}

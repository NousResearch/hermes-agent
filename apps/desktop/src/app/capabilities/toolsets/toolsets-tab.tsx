import './tools.css'

import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useState, type ReactNode } from 'react'

import { SearchField } from '@/components/ui/search-field'
import { Switch } from '@/components/ui/switch'
import { type ProfileScope, setToolsetEnabled } from '@/hermes'
import { useI18n } from '@/i18n'
import { isDesktopToolsetVisible } from '@/lib/desktop-toolsets'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { cn } from '@/lib/utils'
import { $brandSession, capabilitiesUnlocked } from '@/store/brand-session'
import { notify, notifyError } from '@/store/notifications'
import type { ToolsetInfo } from '@/types/hermes'

import { RowButton } from '@/components/ui/row-button'

import { asText, toolNames, toolsetDisplayLabel } from '../../settings/helpers'
import { brandingLockedSlice, FREE_LOCK_MESSAGE, freeListAllowed } from '../free-tier'
import { LockFog, useLockFogTop } from '../lock-fog'
import { CapabilityEmpty } from '../primitives'

import { useToolCalls } from './tool-calls'
import { ToolsetDetail } from './toolset-detail'
import { filteredToolsets, TOOLSETS_QUERY_KEY, toolsetsQueryKey } from './toolsets-data'

const $toolsetsSortDesc = persistentAtom('hermes.desktop.capabilities.toolsetsSortDesc', true, Codecs.bool)

interface ToolsetsTabProps {
  toolsets: ToolsetInfo[]
  profile: ProfileScope
  query: string
  onQueryChange?: (value: string) => void
}

function ToolsRow({
  active,
  busy,
  enabled,
  locked,
  meta,
  onSelect,
  onToggle,
  subtitle,
  title,
  toggleLabel
}: {
  active: boolean
  busy?: boolean
  enabled: boolean
  locked?: boolean
  meta: ReactNode
  onSelect: () => void
  onToggle?: (checked: boolean) => void
  subtitle?: string
  title: string
  toggleLabel?: string
}) {
  return (
    <div
      className={cn(
        'flex w-full items-center justify-between gap-3 px-4 py-3.5',
        locked && 'pointer-events-none opacity-60'
      )}
      data-active={active ? '' : undefined}
      data-tools-row
    >
      <RowButton className="min-w-0 flex-1 text-left" onClick={onSelect}>
        <span className="block truncate text-[15px] font-semibold text-[#0f172a] dark:text-foreground">{title}</span>
        {subtitle ? (
          <span className="mt-1 line-clamp-2 text-[13px] leading-normal text-[#475569]">{subtitle}</span>
        ) : null}
      </RowButton>
      <div className="flex shrink-0 items-center gap-3">
        <span className="whitespace-nowrap text-[12px] font-medium text-[#64748b]">{meta}</span>
        {onToggle ? (
          <Switch
            aria-label={toggleLabel ?? title}
            checked={enabled}
            className="h-[22px] w-10 shrink-0 data-[state=checked]:bg-[#7c3aed] data-[state=checked]:border-transparent [&_[data-slot=switch-thumb]]:size-[18px] data-[state=checked]:[&_[data-slot=switch-thumb]]:translate-x-[18px]"
            disabled={busy || locked}
            onCheckedChange={onToggle}
            size="default"
          />
        ) : null}
      </div>
    </div>
  )
}

/** Tools tab — Figma node 315:3972 marketplace layout. */
export function ToolsetsTab({ onQueryChange, profile, query, toolsets }: ToolsetsTabProps) {
  const { t } = useI18n()
  const toolsetsSortDesc = useStore($toolsetsSortDesc)
  const toolCalls = useToolCalls(profile)
  const [selectedToolset, setSelectedToolset] = useState<string | null>(null)

  const setToolsets = useCallback(
    (fn: (cur: ToolsetInfo[] | undefined) => ToolsetInfo[] | undefined) =>
      queryClient.setQueryData<ToolsetInfo[]>(toolsetsQueryKey(profile), prev => fn(prev) ?? prev),
    [profile]
  )

  const refreshToolsets = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: TOOLSETS_QUERY_KEY })
  }, [])

  const visibleToolsets = useMemo(
    () => filteredToolsets(toolsets, query, toolCalls ?? {}, toolsetsSortDesc),
    [query, toolCalls, toolsets, toolsetsSortDesc]
  )

  const unlocked = capabilitiesUnlocked(useStore($brandSession))
  const allNames = useMemo(() => toolsets.map(row => row.name), [toolsets])
  const isRowLocked = useCallback(
    (name: string) => !freeListAllowed(allNames, name, unlocked),
    [allNames, unlocked]
  )
  const freeToolsets = useMemo(
    () => (unlocked ? visibleToolsets : visibleToolsets.filter(ts => !isRowLocked(ts.name))),
    [isRowLocked, unlocked, visibleToolsets]
  )
  const lockedToolsets = useMemo(
    () => (unlocked ? [] : visibleToolsets.filter(ts => isRowLocked(ts.name))),
    [isRowLocked, unlocked, visibleToolsets]
  )
  const lockedBehind = lockedToolsets.length > 0
  const { freeAnchor, fogTop } = useLockFogTop(lockedBehind, `${freeToolsets.length}:${query}`)
  const selectableToolsets = lockedBehind ? freeToolsets : visibleToolsets

  const activeToolset = useMemo(
    () => selectableToolsets.find(ts => ts.name === selectedToolset) ?? selectableToolsets[0] ?? null,
    [selectedToolset, selectableToolsets]
  )

  const totalVisible = toolsets.filter(ts => isDesktopToolsetVisible(ts.name)).length

  async function handleToggleToolset(toolset: ToolsetInfo, enabled: boolean) {
    if (enabled && isRowLocked(toolset.name)) {
      notify({ kind: 'info', title: 'Locked', message: FREE_LOCK_MESSAGE })
      return
    }

    setToolsets(
      current =>
        current?.map(row => (row.name === toolset.name ? { ...row, enabled, available: enabled } : row)) ?? current
    )

    try {
      await setToolsetEnabled(toolset.name, enabled, profile)
      invalidateSlashCompletions()
    } catch (err) {
      setToolsets(
        current =>
          current?.map(row => (row.name === toolset.name ? { ...row, enabled: !enabled, available: !enabled } : row)) ??
          current
      )
      notifyError(err, t.skills.failedToUpdate(toolsetDisplayLabel(toolset)))
    }
  }

  if (visibleToolsets.length === 0) {
    return (
      <div className="@container/tools flex h-full min-h-0 flex-col" data-tools-page>
        <ToolsPageHeader onQueryChange={onQueryChange} query={query} />
        <CapabilityEmpty noun="tools" query={query} />
      </div>
    )
  }

  const renderRow = (toolset: ToolsetInfo, locked = false) => {
    const label = toolsetDisplayLabel(toolset)

    return (
      <ToolsRow
        active={!locked && activeToolset?.name === toolset.name}
        enabled={toolset.enabled}
        key={toolset.name}
        locked={locked}
        meta={t.skills.toolsCount(toolNames(toolset).length)}
        onSelect={() => {
          if (!locked) {
            setSelectedToolset(toolset.name)
          }
        }}
        onToggle={locked ? undefined : checked => void handleToggleToolset(toolset, checked)}
        subtitle={asText(toolset.description)}
        title={label}
        toggleLabel={t.skills.toggleToolset(label, !toolset.enabled)}
      />
    )
  }

  return (
    <div className="@container/tools flex h-full min-h-0 flex-col" data-tools-page>
      <ToolsPageHeader onQueryChange={onQueryChange} query={query} />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-5 px-7 pb-5 @[64rem]/tools:flex-row">
        <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden" data-tools-list>
          <header className="flex shrink-0 items-center justify-between gap-3 px-4 pb-2 pt-4">
            <button
              className="text-[13px] font-semibold text-[#64748b] hover:text-[#0f172a] dark:hover:text-foreground"
              onClick={() => $toolsetsSortDesc.set(!$toolsetsSortDesc.get())}
              type="button"
            >
              {toolsetsSortDesc ? t.skills.sortMostUsed : t.skills.leastUsed}
            </button>
            <span className="text-[12px] font-medium text-[#64748b]">
              {visibleToolsets.length} of {totalVisible}
            </span>
          </header>

          <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-4 pb-4">
            {lockedBehind ? (
              <div className="relative min-h-full space-y-2">
                <div className="space-y-2" ref={freeAnchor}>
                  {freeToolsets.map(toolset => renderRow(toolset))}
                </div>
                {brandingLockedSlice(lockedToolsets).map(toolset => renderRow(toolset, true))}
                {fogTop > 0 ? (
                  <div className="absolute inset-x-0 bottom-0" style={{ top: fogTop }}>
                    <LockFog label={FREE_LOCK_MESSAGE} />
                  </div>
                ) : null}
              </div>
            ) : (
              visibleToolsets.map(toolset => renderRow(toolset))
            )}
          </div>
        </section>

        <aside
          className="flex min-h-0 w-full shrink-0 flex-col overflow-y-auto @[64rem]/tools:w-[400px]"
          data-tools-detail
        >
          {activeToolset ? (
            <ToolsetDetail
              enabled={activeToolset.enabled}
              onConfiguredChange={refreshToolsets}
              onToggle={checked => void handleToggleToolset(activeToolset, checked)}
              profile={profile}
              toolCalls={toolCalls ?? {}}
              toolset={activeToolset}
            />
          ) : null}
        </aside>
      </div>
    </div>
  )
}

function ToolsPageHeader({
  onQueryChange,
  query
}: {
  onQueryChange?: (value: string) => void
  query: string
}) {
  const { t } = useI18n()

  return (
    <header className="shrink-0 space-y-4 px-7 pb-4 pt-5">
      <div className="min-w-0 space-y-2">
        <h1 className="text-[40px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
          {t.skills.tabToolsets}
        </h1>
        <p className="max-w-2xl text-[15px] text-[#475569] dark:text-(--ui-text-tertiary)">
          {t.skills.toolsPageSubtitle}
        </p>
      </div>
      <SearchField
        containerClassName="w-full min-w-0"
        onChange={value => onQueryChange?.(value)}
        placeholder={t.skills.searchTools}
        shortcutHint={t.catalog.searchShortcut}
        value={query}
        variant="box"
      />
    </header>
  )
}

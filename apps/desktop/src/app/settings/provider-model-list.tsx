import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { Brain, Eye, Layers3, Search, SlidersHorizontal, Zap, type IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { normalize } from '@/lib/text'
import { collapseModelFamilies } from '@/store/model-visibility'
import type { ModelOptionProvider } from '@/types/hermes'

import {
  CAPABILITY_ACTIVE_CLASS,
  CAPABILITY_CHIP_CLASS,
  CAPABILITY_ORDER,
  discoverButtonLabel,
  groupModelsByLetter,
  inferModelCapabilities,
  sortModelFamilies,
  type CapabilityFilter,
  type ModelCapabilities,
  type ModelSortMode
} from './model-list-utils'
import { useProviderModelVisibility } from './use-provider-visibility'

interface ProviderModelListProps {
  enabled?: boolean
  provider: ModelOptionProvider
  /** Trigger model discovery (custom) or catalog refresh (built-in). */
  onDiscover?: () => void | Promise<void>
  /** Custom providers only: open the manual add-model dialog. */
  onAddModel?: () => void
  /** True while a discover/refresh RPC is in flight — disables the button. */
  discoverWorking?: boolean
}

/** Decorative SVG icon per capability filter chip. */
const CAPABILITY_ICON: Record<CapabilityFilter, IconComponent> = {
  vision: Eye,
  multimodal: Layers3,
  reasoning: Brain,
  fast: Zap
}

/** Outlined search field with a leading search icon. Owns no state — the parent
 *  passes value/onChange so the toolbar can share a row with the filter chips.
 *  Hover darkens the border; focus adds an accent border + soft ring and tints
 *  the icon, so the cursor position is unmistakable. */
function SearchField({
  onChange,
  placeholder,
  value
}: {
  onChange: (value: string) => void
  placeholder: string
  value: string
}) {
  return (
    <div
      className={cn(
        'group flex h-7 min-w-[10rem] flex-1 items-center gap-1.5 rounded-md border bg-(--ui-bg-tertiary) px-2',
        'border-(--ui-stroke-tertiary) transition-[border-color,box-shadow,background-color] duration-150',
        'hover:border-(--ui-stroke-secondary)',
        'focus-within:border-(--ui-accent) focus-within:bg-(--ui-bg) focus-within:ring-2 focus-within:ring-(--ui-accent)/25'
      )}
    >
      <Search className="size-3.5 shrink-0 text-(--ui-text-tertiary) transition-colors group-focus-within:text-(--ui-accent)" />
      <input
        aria-label={placeholder}
        className="min-w-0 flex-1 bg-transparent text-xs text-foreground placeholder:text-(--ui-text-tertiary) focus:outline-none"
        onChange={event => onChange(event.target.value)}
        placeholder={placeholder}
        type="text"
        value={value}
      />
    </div>
  )
}

interface ModelRowProps {
  provider: ModelOptionProvider
  family: { id: string }
  displayName: string
  checked: boolean
  enabled: boolean
  caps: ModelCapabilities
  pricing: { input: string; output: string; cache: unknown; free: boolean } | undefined
  toggle: (id: string) => void
}

/** Single model row: name (bold when active), capability badges, pricing, switch. */
function ModelRow({ provider, family, displayName, checked, enabled, caps, pricing, toggle }: ModelRowProps) {
  const { t } = useI18n()
  const copy = t.providerManager
  const unavailable = provider.unavailable_models?.includes(family.id)

  return (
    <label
      className={cn(
        'flex cursor-pointer items-center gap-2 px-2 py-1 text-sm hover:bg-accent/50',
        checked && 'font-bold',
        unavailable && 'cursor-not-allowed opacity-60'
      )}
      key={family.id}
    >
      <span className="min-w-0 flex-1 truncate">
        {displayName}
        {CAPABILITY_ORDER.filter(key => caps[key]).map(key => (
          <span
            className={cn(
              'ml-1 inline-block rounded px-1 py-px text-[0.5625rem] font-medium uppercase tracking-wide',
              CAPABILITY_ACTIVE_CLASS[key]
            )}
            key={key}
          >
            {copy[`filter${key[0].toUpperCase()}${key.slice(1)}` as 'filterVision']}
          </span>
        ))}
        {pricing ? (
          <span className="ml-1 text-[0.6875rem] text-(--ui-text-tertiary)">
            {pricing.free ? 'free' : `${pricing.input} / ${pricing.output}`}
          </span>
        ) : null}
        {unavailable ? (
          <span className="ml-1 text-[0.6875rem] text-(--ui-text-tertiary)">· {copy.unavailable}</span>
        ) : null}
      </span>
      <Switch
        aria-label={displayName}
        checked={checked}
        disabled={!enabled || unavailable}
        onCheckedChange={() => toggle(family.id)}
      />
    </label>
  )
}

/**
 * Right pane of the Provider Manager: every model of the selected provider as
 * an active/inactive toggle. Reuses the load-bearing visibility store via
 * useProviderModelVisibility (no fork of the sentinel logic) so a hidden
 * provider and a never-customized one behave identically to the rest of the
 * app. When the provider is deactivated (`enabled === false`), every model
 * reads as hidden and toggling is disabled.
 *
 * Enhancements over the base list:
 *  - inline alphabetical letter-group headers (B1)
 *  - capability filter chips (B2)
 *  - sort mode (active-first / A→Z / Z→A) (B3)
 *  - "active only" toggle + bulk activate/deactivate all (B4)
 *  - pricing + unavailable-model surfacing (B5)
 */
export function ProviderModelList({
  enabled = true,
  provider,
  onDiscover,
  onAddModel,
  discoverWorking = false
}: ProviderModelListProps) {
  const { t } = useI18n()
  const copy = t.providerManager
  const { isVisible, toggle, setMany, allHidden, visibleCount } = useProviderModelVisibility(provider.slug, [provider], enabled)

  const [search, setSearch] = useState('')
  const [capFilters, setCapFilters] = useState<Record<CapabilityFilter, boolean>>({
    vision: false,
    multimodal: false,
    reasoning: false,
    fast: false
  })
  const [sortMode, setSortMode] = useState<ModelSortMode>('activeFirst')
  const [activeOnly, setActiveOnly] = useState(false)
  const [groupByLetter, setGroupByLetter] = useState(false)

  const q = normalize(search)

  const families = useMemo(() => collapseModelFamilies(provider.models ?? []), [provider.models])

  // Capability inference per family (backend capabilities when present).
  const capabilitiesFor = (familyId: string) => {
    const display = provider.model_display_names?.[familyId] ?? displayModelName(familyId)
    const backend = provider.capabilities?.[familyId]
    return inferModelCapabilities(familyId, display, backend ? { reasoning: backend.reasoning, fast: backend.fast } : undefined)
  }

  const matchesSearch = (familyId: string) =>
    !q || `${familyId} ${provider.name} ${provider.slug} ${displayModelName(familyId)}`.toLowerCase().includes(q)

  const matchesCaps = (familyId: string) => {
    const caps = capabilitiesFor(familyId)
    return (Object.keys(capFilters) as CapabilityFilter[]).every(key => !capFilters[key] || caps[key])
  }

  const filtered = families.filter(family => matchesSearch(family.id) && matchesCaps(family.id))

  // Non-destructive sort: the "active on top" promotion must NOT re-run on
  // every toggle (that makes an activated model instantly jump to the top,
  // which is disorienting). We freeze the active-state snapshot and only
  // refresh it when the model *set* or sort mode changes (a list update),
  // exactly as requested. Toggles still flip the switch immediately; the
  // reorder happens on the next list refresh.
  const listKey = families.map(f => f.id).join('|') + '::' + sortMode
  const frozenVisible = useMemo(
    () => new Set(families.map(f => f.id).filter(id => isVisible(id))),
    // Intentionally excludes `isVisible`: we want a stable snapshot that does
    // not change when a single model is toggled.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [listKey]
  )

  const sorted = useMemo(
    () => sortModelFamilies(filtered, sortMode, id => frozenVisible.has(id)),
    [filtered, sortMode, frozenVisible]
  )

  const visible = activeOnly ? sorted.filter(family => isVisible(family.id)) : sorted

  const grouped = useMemo(() => groupModelsByLetter(visible), [visible])

  const allModelIds = families.map(f => f.id)
  const total = allModelIds.length

  const toggleCap = (key: CapabilityFilter) => setCapFilters(prev => ({ ...prev, [key]: !prev[key] }))
  const activeCapCount = CAPABILITY_ORDER.filter(key => capFilters[key]).length

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between gap-2 px-3 pb-1 pt-2">
        <div className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
          {copy.modelsHeading}
        </div>
        <div className="flex items-center gap-2">
          <Button
            disabled={!enabled || discoverWorking}
            onClick={() => void onDiscover?.()}
            size="sm"
            type="button"
            variant="outline"
          >
            {discoverButtonLabel(families.length, copy)}
          </Button>
          {provider.is_user_defined && (
            <Button
              disabled={!enabled}
              onClick={() => onAddModel?.()}
              size="sm"
              type="button"
              variant="outline"
            >
              {copy.addModel}
            </Button>
          )}
          <div className="text-[0.6875rem] text-(--ui-text-tertiary)">{copy.activeOfTotal(visibleCount, total)}</div>
        </div>
      </div>

      {/* Toolbar: search field (left) + capability filter chips (right).
          The row is a container query context: when it's too narrow for the
          inline chips they collapse into a single "Filters" dropdown. Both
          surfaces drive the same capFilters state so they never drift. */}
      <div className="@container flex items-center gap-2 px-3 pb-1.5 pt-1" data-testid="model-toolbar">
        <SearchField onChange={setSearch} placeholder={copy.searchPlaceholder} value={search} />

        {/* Inline chips — hidden below the container breakpoint. */}
        <div className="hidden shrink-0 items-center gap-1.5 @[24rem]:flex">
          {CAPABILITY_ORDER.map(key => {
            const Icon = CAPABILITY_ICON[key]
            return (
              <button
                aria-pressed={capFilters[key]}
                className={cn(
                  'flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.6875rem] transition-colors',
                  capFilters[key] ? CAPABILITY_ACTIVE_CLASS[key] : CAPABILITY_CHIP_CLASS[key]
                )}
                key={key}
                onClick={() => toggleCap(key)}
                type="button"
              >
                <Icon className="size-3 shrink-0" />
                {copy[`filter${key[0].toUpperCase()}${key.slice(1)}` as 'filterVision']}
              </button>
            )
          })}
        </div>

        {/* Collapsed "Filters" dropdown — shown below the container breakpoint. */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className={cn(
                'flex shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[0.6875rem] transition-colors @[24rem]:hidden',
                activeCapCount > 0
                  ? 'border-(--ui-accent)/40 bg-(--ui-accent)/15 text-foreground'
                  : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-(--ui-text-secondary) hover:text-foreground'
              )}
              type="button"
            >
              <SlidersHorizontal className="size-3 shrink-0" />
              {copy.filters}
              {activeCapCount > 0 && (
                <span className="ml-0.5 rounded-full bg-(--ui-accent) px-1 text-[0.5625rem] font-semibold text-(--ui-accent-foreground)">
                  {activeCapCount}
                </span>
              )}
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-40">
            {CAPABILITY_ORDER.map(key => {
              const Icon = CAPABILITY_ICON[key]
              return (
                <DropdownMenuCheckboxItem
                  checked={capFilters[key]}
                  key={key}
                  onCheckedChange={() => toggleCap(key)}
                >
                  <Icon className={cn('size-3.5', capFilters[key] ? CAPABILITY_ACTIVE_CLASS[key].match(/text-\S+/)?.[0] : 'text-(--ui-text-tertiary)')} />
                  {copy[`filter${key[0].toUpperCase()}${key.slice(1)}` as 'filterVision']}
                </DropdownMenuCheckboxItem>
              )
            })}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Secondary controls: group-by-letter + sort */}
      <div className="flex items-center gap-1.5 px-3 pb-1.5">
        <button
          aria-pressed={groupByLetter}
          className={cn(
            'rounded-full px-2 py-0.5 text-[0.6875rem] transition-colors',
            groupByLetter
              ? 'bg-(--ui-accent) text-(--ui-accent-foreground)'
              : 'bg-(--ui-bg-tertiary) text-(--ui-text-tertiary) hover:text-foreground'
          )}
          onClick={() => setGroupByLetter(!groupByLetter)}
          type="button"
        >
          {copy.groupByLetter}
        </button>
        <select
          aria-label={copy.sortActiveFirst}
          className="ml-auto h-6 rounded bg-(--ui-bg-tertiary) px-1 text-[0.6875rem] text-foreground focus:outline-none"
          onChange={event => setSortMode(event.target.value as ModelSortMode)}
          value={sortMode}
        >
          <option value="activeFirst">{copy.sortActiveFirst}</option>
          <option value="az">{copy.sortAz}</option>
          <option value="za">{copy.sortZa}</option>
        </select>
      </div>

      {/* Bulk actions + active-only */}
      <div className="flex items-center gap-2 px-3 pb-1.5">
        <label className="flex items-center gap-1.5 text-[0.6875rem] text-(--ui-text-secondary)">
          <input
            checked={activeOnly}
            onChange={event => setActiveOnly(event.target.checked)}
            type="checkbox"
          />
          {copy.activeOnly}
        </label>
        <div className="ml-auto flex items-center gap-1.5">
          <Button
            disabled={!enabled || total === 0}
            onClick={() => setMany(allModelIds, true)}
            size="sm"
            type="button"
            variant="ghost"
          >
            {copy.activateAll}
          </Button>
          <Button
            disabled={!enabled || total === 0}
            onClick={() => setMany(allModelIds, false)}
            size="sm"
            type="button"
            variant="ghost"
          >
            {copy.deactivateAll}
          </Button>
        </div>
      </div>

      {allHidden && (
        <div className="mx-3 mb-1 rounded bg-(--ui-bg-tertiary) px-2 py-1 text-[0.6875rem] text-(--ui-text-tertiary)">
          {copy.allHidden}
        </div>
      )}

      {!enabled && (
        <div className="mx-3 mb-1 rounded border border-(--ui-stroke-tertiary) px-2 py-1 text-[0.6875rem] text-(--ui-text-tertiary)">
          {copy.disableProvider}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {visible.length === 0 ? (
          <div className="px-3 py-5 text-center text-xs text-muted-foreground">{copy.noModels}</div>
        ) : groupByLetter ? (
          grouped.map(group => (
            <div key={group.letter}>
              <div
                aria-hidden="true"
                className="sticky top-0 bg-(--ui-bg) px-2 py-0.5 text-[0.625rem] font-semibold uppercase tracking-wide text-(--ui-text-tertiary)"
              >
                {group.letter}
              </div>
              {group.families.map(family => (
                <ModelRow
                  caps={capabilitiesFor(family.id)}
                  checked={isVisible(family.id)}
                  displayName={provider.model_display_names?.[family.id] ?? modelDisplayParts(family.id).name}
                  enabled={enabled}
                  family={family}
                  key={family.id}
                  pricing={provider.pricing?.[family.id]}
                  provider={provider}
                  toggle={toggle}
                />
              ))}
            </div>
          ))
        ) : (
          visible.map(family => (
            <ModelRow
              caps={capabilitiesFor(family.id)}
              checked={isVisible(family.id)}
              displayName={provider.model_display_names?.[family.id] ?? modelDisplayParts(family.id).name}
              enabled={enabled}
              family={family}
              key={family.id}
              pricing={provider.pricing?.[family.id]}
              provider={provider}
              toggle={toggle}
            />
          ))
        )}
      </div>
    </div>
  )
}

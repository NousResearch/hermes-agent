import { type KeyboardEvent } from 'react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { ModelOptionProvider } from '@/types/hermes'

import { flattenGroups, groupProviders, type ProviderGroupId } from './provider-grouping'
import { $visibleModels, providerActiveCounts } from '@/store/model-visibility'
import { useStore } from '@nanostores/react'

interface ProviderManagerNavProps {
  onAdd: () => void
  onSelect: (slug: string) => void
  providers: ModelOptionProvider[]
  selectedSlug: string | null
  /** Provider-list search query (controlled by the parent container). */
  providerSearch?: string
  /** Called when the provider-list search input changes. */
  onProviderSearch?: (value: string) => void
}

const GROUP_LABEL: Record<ProviderGroupId, keyof ReturnType<typeof useGroupCopy>> = {
  local: 'groupLocal',
  configured: 'groupConfigured',
  unconfigured: 'groupUnconfigured'
}

// Per-group header color so it reads at a glance which providers are usable
// (active → green) versus still needing setup (unconfigured → muted). The
// `local` group is self-hosted and active, so it shares the active color.
const GROUP_HEADER_CLASS: Record<ProviderGroupId, string> = {
  local: 'text-(--ui-green)',
  configured: 'text-(--ui-green)',
  unconfigured: 'text-(--ui-text-tertiary)'
}

// Small indirection so the label lookup stays typed against the i18n block.
function useGroupCopy() {
  const { t } = useI18n()
  return {
    groupLocal: t.providerManager.groupLocal,
    groupConfigured: t.providerManager.groupConfigured,
    groupUnconfigured: t.providerManager.groupUnconfigured
  }
}

/**
 * Left pane of the Provider Manager: a selectable, keyboard-navigable list of
 * providers grouped into Local / Configured / Unconfigured. Selection drives
 * the right pane (ProviderModelList). Implemented as an ARIA listbox so arrow
 * keys move between providers and screen readers announce the active one.
 * Disabled providers are dimmed; the activation toggle itself lives in the
 * right-pane header (keeps the listbox ARIA-clean). Group headers are visual
 * only (aria-hidden) — keyboard navigation uses the flattened provider order.
 */
export function ProviderManagerNav({
  onAdd,
  onSelect,
  providers,
  selectedSlug,
  providerSearch = '',
  onProviderSearch
}: ProviderManagerNavProps) {
  const { t } = useI18n()
  const copy = t.providerManager
  const groupLabels = useGroupCopy()

  const groups = groupProviders(providers)
  const flat = flattenGroups(groups)

  // Active/total model counts per provider, derived from the shared visibility
  // store. Recomputed when the store or the provider list changes.
  const stored = useStore($visibleModels)
  const counts = providerActiveCounts(stored, providers)

  const onKeyDown = (event: KeyboardEvent<HTMLUListElement>) => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
      return
    }

    const idx = flat.findIndex(provider => provider.slug === selectedSlug)
    const nextIdx =
      event.key === 'ArrowDown' ? Math.min(flat.length - 1, idx + 1) : Math.max(0, idx - 1)

    if (nextIdx >= 0 && nextIdx < flat.length) {
      event.preventDefault()
      onSelect(flat[nextIdx].slug)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-3 pb-1 pt-2">
        <span className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
          {copy.providerListHeading}
        </span>
        <button
          className="rounded px-1.5 py-0.5 text-[0.6875rem] text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground"
          onClick={onAdd}
          type="button"
        >
          {copy.addProvider}
        </button>
      </div>
      {onProviderSearch && (
        <div className="px-2 pb-1.5">
          <input
            aria-label={copy.searchProviders}
            className="h-6 w-full rounded bg-(--ui-bg-tertiary) px-2 text-xs text-foreground placeholder:text-(--ui-text-tertiary) focus:outline-none"
            onChange={event => onProviderSearch(event.target.value)}
            placeholder={copy.searchProviders}
            type="text"
            value={providerSearch}
          />
        </div>
      )}
      {flat.length === 0 ? (
        <div className="px-3 py-5 text-center text-xs text-muted-foreground">{copy.noProvidersMatch}</div>
      ) : (
        <ul
          aria-label={copy.providerListHeading}
          className="min-h-0 flex-1 overflow-y-auto px-2 pb-2"
          onKeyDown={onKeyDown}
          role="listbox"
        >
          {groups.map((group, index) => (
          <li key={group.id} role="presentation">
            {/* Section header: a top divider + larger gap separates each group so
                the rows below clearly belong to it. Active groups (Local /
                Configured) use the green accent; Unconfigured stays muted so it
                reads as "not set up yet". The first group skips the divider. */}
            <div
              aria-hidden="true"
              className={cn(
                'px-2 pb-1 text-[0.625rem] font-semibold uppercase tracking-wide',
                GROUP_HEADER_CLASS[group.id],
                index === 0 ? 'mt-1' : 'mt-3 border-t border-(--ui-stroke-tertiary) pt-2'
              )}
            >
              {groupLabels[GROUP_LABEL[group.id]]}
            </div>
            <ul role="presentation" className="flex flex-col gap-px">
              {group.providers.map(provider => {
                const selected = provider.slug === selectedSlug
                const disabled = provider.enabled === false
                const count = counts[provider.slug] ?? { active: 0, total: provider.models?.length ?? 0 }
                const unconfigured = provider.authenticated === false && !provider.is_user_defined

                return (
                  <li key={provider.slug} role="option" aria-selected={selected}>
                    <button
                      aria-current={selected ? 'true' : undefined}
                      className={cn(
                        'flex h-8 w-full items-center justify-between gap-2 rounded-md px-2 text-left text-sm transition-colors',
                        selected
                          ? 'bg-(--ui-bg-tertiary) text-foreground'
                          : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground',
                        disabled && 'opacity-50'
                      )}
                      onClick={() => onSelect(provider.slug)}
                      tabIndex={selected ? 0 : -1}
                      type="button"
                    >
                      <span className="min-w-0 flex-1 truncate">{provider.name}</span>
                      {disabled ? (
                        <span className="shrink-0 rounded bg-(--ui-bg-tertiary) px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)">
                          {copy.disableProvider}
                        </span>
                      ) : unconfigured ? (
                        <span
                          aria-label={copy.searchProviders}
                          className="shrink-0 rounded bg-(--ui-bg-tertiary) px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)"
                          title={copy.noProvidersMatch}
                        >
                          ⚠
                        </span>
                      ) : (
                        <span
                          className={cn(
                            'shrink-0 rounded px-1.5 text-[0.6875rem] tabular-nums',
                            selected
                              ? 'bg-(--ui-stroke-tertiary) text-foreground'
                              : 'bg-(--ui-bg-tertiary) text-(--ui-text-tertiary)'
                          )}
                        >
                          {copy.activeOfTotal(count.active, count.total)}
                        </span>
                      )}
                    </button>
                  </li>
                )
              })}
            </ul>
          </li>
        ))}
        </ul>
      )}
    </div>
  )
}

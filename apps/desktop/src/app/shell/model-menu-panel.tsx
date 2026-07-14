import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { createContext, useContext, useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  dropdownMenuRow,
  DropdownMenuSearch,
  dropdownMenuSectionLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { Skeleton } from '@/components/ui/skeleton'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { requestModelOptions } from '@/lib/model-options'
import {
  currentPickerSelection,
  displayModelName,
  modelDisplayParts,
  reasoningEffortLabel
} from '@/lib/model-status-label'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { $modelPresets, applyModelPreset, modelPresetKey } from '@/store/model-presets'
import {
  $visibleModels,
  collapseModelFamilies,
  DEFAULT_VISIBLE_PER_PROVIDER,
  effectiveVisibleKeys,
  type ModelFamily,
  modelVisibilityKey,
  setModelVisibilityOpen
} from '@/store/model-visibility'
import {
  $activeSessionId,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort
} from '@/store/session'
import type { ModelOptionProvider, ModelOptionsResponse } from '@/types/hermes'

import { ModelEditSubmenu, resolveFastControl } from './model-edit-submenu'

// Lets the host dropdown (model-pill) hand the panel a way to dismiss itself so
// clicking a model row commits + closes, while the hover-revealed edit submenu
// (reasoning/fast) stays open to play with (its items preventDefault on select).
export const ModelMenuCloseContext = createContext<() => void>(() => {})

interface ModelMenuPanelProps {
  gateway?: HermesGateway
  onSelectModel: (selection: { model: string; provider: string }) => Promise<boolean> | void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

interface ProviderGroup {
  families: ModelFamily[]
  provider: ModelOptionProvider
}

export function ModelMenuPanel({ gateway, onSelectModel, requestGateway }: ModelMenuPanelProps) {
  const { t } = useI18n()
  const copy = t.shell.modelMenu
  const closeMenu = useContext(ModelMenuCloseContext)
  const [search, setSearch] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const queryClient = useQueryClient()
  // Reactive session state is read from the stores here (not drilled in), so
  // toggling effort/fast/model re-renders this panel in place without forcing
  // the parent to rebuild the menu content (which would close the dropdown).
  const activeSessionId = useStore($activeSessionId)
  const currentFastMode = useStore($currentFastMode)
  const currentModel = useStore($currentModel)
  const currentProvider = useStore($currentProvider)
  const currentReasoningEffort = useStore($currentReasoningEffort)
  const modelPresets = useStore($modelPresets)
  const visibleModels = useStore($visibleModels)

  const modelOptions = useQuery({
    queryKey: ['model-options', activeSessionId || 'global'],
    // Gateway-first even with no session yet: a connected (possibly remote)
    // gateway owns the model catalog, including virtual providers like `moa`
    // that the local REST fallback can't know about (#53817).
    queryFn: (): Promise<ModelOptionsResponse> => requestModelOptions({ gateway, sessionId: activeSessionId })
  })

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
    !!activeSessionId,
    { model: currentModel, provider: currentProvider },
    modelOptions.data
  )

  const loading = modelOptions.isPending && !modelOptions.data

  const error = modelOptions.error
    ? modelOptions.error instanceof Error
      ? modelOptions.error.message
      : String(modelOptions.error)
    : null

  const providers = modelOptions.data?.providers

  // The catalog carries MoA presets as a virtual `moa` provider row. Render
  // them in their dedicated section below and keep the row out of the main
  // provider groups so presets don't show up twice.
  const moaPresets = useMemo(
    () => providers?.find(provider => provider.slug.toLowerCase() === 'moa')?.models ?? [],
    [providers]
  )

  const pickerProviders = useMemo(
    () => providers?.filter(provider => provider.slug.toLowerCase() !== 'moa') ?? [],
    [providers]
  )

  const effectiveVisibleModels = useMemo(
    () => effectiveVisibleKeys(visibleModels, pickerProviders),
    [visibleModels, pickerProviders]
  )

  // The composer picker never persists the profile default. With a session it
  // scopes the switch to that session; with none it's UI state shipped on the
  // next session.create (see selectModel). The default lives in Settings → Model.
  const switchTo = (model: string, provider: string) => onSelectModel({ model, provider })

  // Explicit "Refresh Models": re-fetch the catalog with refresh:true so the
  // backend busts its 1h provider-model disk cache and re-pulls each provider's
  // live list. Fixes live-only models (e.g. OpenCode Zen free tier) vanishing
  // when the cache expires and falls back to the curated static list.
  const refreshModels = async () => {
    if (refreshing) {
      return
    }

    setRefreshing(true)

    try {
      const queryKey = ['model-options', activeSessionId || 'global']

      const next = await requestModelOptions({ gateway, refresh: true, sessionId: activeSessionId })

      queryClient.setQueryData<ModelOptionsResponse>(queryKey, next)
    } catch {
      // Network/backend hiccup — fall back to a plain invalidate so the next
      // open re-fetches (still cached, but no worse than before).
      void queryClient.invalidateQueries({ queryKey: ['model-options'] })
    } finally {
      setRefreshing(false)
    }
  }

  // Selecting a model row restores that model's remembered preset onto the
  // session (effort/fast), gated by capability. Unset → Hermes defaults.
  const selectFamily = async (family: ModelFamily, provider: ModelOptionProvider) => {
    const caps = provider.capabilities?.[family.id]
    const preset = modelPresets[modelPresetKey(provider.slug, family.id)] ?? {}

    // Variant-fast models (no speed param) express "fast" as a separate `-fast`
    // id, so honor the saved preset by selecting that sibling. Param-fast is
    // applied via applyModelPreset below instead.
    const variantFast = !(caps?.fast ?? false) && !!family.fastId
    const targetId = variantFast && preset.fast === true ? family.fastId! : family.id

    if ((await switchTo(targetId, provider.slug)) === false) {
      return
    }

    await applyModelPreset(
      {
        effort: (caps?.reasoning ?? true) ? (preset.effort ?? 'medium') : undefined,
        fast: (caps?.fast ?? false) ? (preset.fast ?? false) : undefined
      },
      { failMessage: t.shell.modelOptions.updateFailed, request: requestGateway, sessionId: activeSessionId }
    )
  }

  // Selecting a MoA preset switches the session to it PERSISTENTLY, using the
  // same path real provider selections use (onSelectModel → config.set with
  // --session for live sessions → the gateway's persistent switch_model).
  // Previously this dispatched the one-shot `/moa` command, which ran a single
  // turn through MoA and then silently reverted to the prior model (#54670) —
  // the dropdown presented presets like persistent selections but they weren't.
  // No session gate: like regular model rows, a pre-session pick is UI state
  // shipped on the next session.create.
  const selectMoaPreset = async (preset: string) => {
    if ((await switchTo(preset, 'moa')) === false) {
      return
    }

    closeMenu()
  }

  const groups = useMemo(
    () =>
      groupModels(pickerProviders, search, { model: optionsModel, provider: optionsProvider }, effectiveVisibleModels),
    [pickerProviders, search, optionsModel, optionsProvider, effectiveVisibleModels]
  )

  return (
    <>
      <DropdownMenuSearch aria-label={copy.search} onValueChange={setSearch} placeholder={copy.search} value={search} />

      <DropdownMenuSeparator className="mx-0" />

      {loading ? (
        <DropdownMenuGroup className="py-1">
          {Array.from({ length: 4 }, (_, index) => (
            <DropdownMenuItem
              className={dropdownMenuRow}
              disabled
              key={index}
              onSelect={event => event.preventDefault()}
            >
              <Skeleton className="h-4 w-full" />
            </DropdownMenuItem>
          ))}
        </DropdownMenuGroup>
      ) : error ? (
        <DropdownMenuItem className={dropdownMenuRow} disabled>
          {error}
        </DropdownMenuItem>
      ) : groups.length === 0 && moaPresets.length === 0 ? (
        <DropdownMenuItem className={dropdownMenuRow} disabled>
          {copy.noModels}
        </DropdownMenuItem>
      ) : (
        <div className="max-h-[max(150px,30dvh)] overflow-y-auto py-0.5">
          {groups.map(group => (
            <DropdownMenuGroup className="py-0.5" key={group.provider.slug}>
              <DropdownMenuLabel className={dropdownMenuSectionLabel}>{group.provider.name}</DropdownMenuLabel>
              {group.families.map(family => {
                // The active id may be the base or its -fast sibling; either
                // way this one family row represents both.
                const activeId =
                  group.provider.slug === optionsProvider &&
                  (optionsModel === family.id || optionsModel === family.fastId)
                    ? optionsModel
                    : null

                const isCurrent = activeId !== null
                const name = modelDisplayParts(family.id).name
                // Capabilities are looked up against the active/base id; the
                // -fast variant carries the same param support as its base.
                const caps = group.provider.capabilities?.[family.id]

                // Effective settings for this row: live session state when it's
                // the active model, otherwise its remembered preset (Hermes
                // defaults when unset). Row label AND submenu read from these so
                // they never disagree.
                const preset = modelPresets[modelPresetKey(group.provider.slug, family.id)] ?? {}
                const effEffort = isCurrent ? currentReasoningEffort : (preset.effort ?? '')
                const effFast = isCurrent ? currentFastMode : (preset.fast ?? false)

                const fastControl = resolveFastControl(
                  activeId ?? family.id,
                  group.provider.models ?? [],
                  caps?.fast ?? false,
                  effFast
                )

                const meta = [
                  fastControl.kind !== 'none' && fastControl.on ? copy.fast : null,
                  (caps?.reasoning ?? true) ? reasoningEffortLabel(effEffort) || copy.medium : null
                ]
                  .filter(Boolean)
                  .join(' ')

                // Every row is a hover-Edit submenu trigger. Activating it
                // (pointer or keyboard) switches to the family's base model and
                // restores its preset; the Fast toggle inside swaps to the -fast
                // sibling (or flips the speed param). The sub-trigger has no
                // `onSelect`, so wire both click and Enter/Space for keyboard parity.
                // Clicking the row commits the model and closes the picker; the
                // edit submenu (reasoning/fast) is reached by HOVER, so you can
                // still tweak those without the click dismissing everything.
                const activate = () => {
                  if (!isCurrent) {
                    void selectFamily(family, group.provider)
                  }

                  closeMenu()
                }

                return (
                  <DropdownMenuSub key={`${group.provider.slug}:${family.id}`}>
                    <DropdownMenuSubTrigger
                      className={dropdownMenuRow}
                      hideChevron
                      onClick={activate}
                      onKeyDown={event => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          activate()
                        }
                      }}
                    >
                      <span className="min-w-0 flex-1 truncate">
                        {name}
                        {meta ? <span className="text-(--ui-text-tertiary)"> {meta}</span> : null}
                      </span>
                      {isCurrent ? <Codicon className="ml-auto text-foreground" name="check" size="0.75rem" /> : null}
                    </DropdownMenuSubTrigger>
                    <ModelEditSubmenu
                      effort={effEffort}
                      fastControl={fastControl}
                      isActive={isCurrent}
                      model={family.id}
                      onSelectModel={nextModel => switchTo(nextModel, group.provider.slug)}
                      provider={group.provider.slug}
                      reasoning={caps?.reasoning ?? true}
                      requestGateway={requestGateway}
                    />
                  </DropdownMenuSub>
                )
              })}
            </DropdownMenuGroup>
          ))}
        </div>
      )}

      <DropdownMenuSeparator className="mx-0" />

      {moaPresets.length > 0 ? (
        <>
          <DropdownMenuLabel className={dropdownMenuSectionLabel}>MoA presets</DropdownMenuLabel>
          {moaPresets.map(preset => {
            const isCurrentMoa = optionsProvider === 'moa' && optionsModel === preset

            return (
              <DropdownMenuItem
                className={dropdownMenuRow}
                key={`moa:${preset}`}
                onSelect={event => {
                  event.preventDefault()
                  void selectMoaPreset(preset)
                }}
              >
                <span className="min-w-0 flex-1 truncate">MoA: {preset}</span>
                {isCurrentMoa ? <Codicon className="ml-auto text-foreground" name="check" size="0.75rem" /> : null}
              </DropdownMenuItem>
            )
          })}
          <DropdownMenuSeparator className="mx-0" />
        </>
      ) : null}

      <DropdownMenuItem
        className={cn(dropdownMenuRow, 'text-(--ui-text-tertiary)')}
        disabled={refreshing}
        onSelect={event => {
          event.preventDefault()
          void refreshModels()
        }}
      >
        <Codicon className={cn(refreshing && 'animate-spin')} name="sync" size="0.75rem" />
        {copy.refreshModels}
      </DropdownMenuItem>

      <DropdownMenuItem
        className={cn(dropdownMenuRow, 'text-(--ui-text-tertiary)')}
        onSelect={() => setModelVisibilityOpen(true)}
      >
        <Codicon name="settings-gear" size="0.75rem" />
        {copy.editModels}
      </DropdownMenuItem>
    </>
  )
}

// Collapsed we show the user's chosen models (or the curated default); typing
// spans every available model so anything is reachable past the cut. A search
// is itself a narrowing action, so we do NOT cap per-provider matches — a
// provider serving 19 models (e.g. opencode-go) must show all 19 when the user
// searches for it, not a truncated subset. (#47077 follow-up)

function groupModels(
  providers: ModelOptionProvider[],
  search: string,
  current: { model: string; provider: string },
  visible: Set<string> | null
): ProviderGroup[] {
  const q = normalize(search)
  const groups: ProviderGroup[] = []

  for (const provider of providers) {
    const allFamilies = collapseModelFamilies(provider.models ?? [])

    if (allFamilies.length === 0) {
      continue
    }

    const matches = (family: ModelFamily) =>
      `${family.id} ${family.fastId ?? ''} ${provider.name} ${provider.slug} ${displayModelName(family.id)}`
        .toLowerCase()
        .includes(q)

    // Which model ids to show (the active one is always added on top of this).
    let shown: Set<string>

    if (q) {
      // Search spans every family, regardless of visibility.
      shown = new Set(allFamilies.filter(matches).map(family => family.id))
    } else if (visible) {
      // User has customized which models show — honor their selection exactly.
      shown = new Set(
        allFamilies.filter(family => visible.has(modelVisibilityKey(provider.slug, family.id))).map(family => family.id)
      )
    } else {
      // Default: curated top-N families per provider.
      shown = new Set(allFamilies.slice(0, DEFAULT_VISIBLE_PER_PROVIDER).map(family => family.id))
    }

    // Always include the active model — but keep every row in the provider's
    // stable curated order (filter `allFamilies`, never reorder), so selecting
    // a model can't shuffle the list.
    const activeId =
      provider.slug === current.provider && current.model
        ? allFamilies.find(family => family.id === current.model || family.fastId === current.model)?.id
        : undefined

    const families = allFamilies.filter(family => shown.has(family.id) || family.id === activeId)

    if (families.length > 0) {
      groups.push({ families, provider })
    }
  }

  // Stable, logical group order: alphabetical by provider name. (The backend
  // floats the current provider first, which would reshuffle on every switch.)
  groups.sort((a, b) => a.provider.name.localeCompare(b.provider.name))

  return groups
}

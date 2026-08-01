import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
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
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { usePointerQuiet } from '@/components/ui/keyboard-first'
import { Skeleton } from '@/components/ui/skeleton'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection, displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { DEFAULT_REASONING_EFFORT, reasoningEffortLabel } from '@/lib/reasoning-effort'
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
import { $collapsedProviders, toggleCollapsedProvider } from '@/store/provider-collapse'
import { $defaultReasoningEffort } from '@/store/session'
import type { ModelOptionProvider, ModelOptionsResponse } from '@/types/hermes'

import { ModelEditSubmenu, resolveFastControl } from './model-edit-submenu'

// Lets the host dropdown (model-pill) hand the panel a way to dismiss itself so
// clicking a model row commits + closes, while the hover-revealed edit submenu
// (reasoning/fast) stays open to play with (its items preventDefault on select).
export const ModelMenuCloseContext = createContext<() => void>(() => {})

export interface ModelSelection {
  model: string
  provider: string
  /** Runtime id of the surface that opened the menu. When set, the switch
   *  targets that session (a tile) instead of the primary `$activeSessionId`. */
  sessionId?: null | string
}

interface ModelMenuPanelProps {
  gateway?: HermesGateway
  onSelectModel: (selection: ModelSelection) => Promise<boolean> | void
  profile?: string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

interface ProviderGroup {
  families: ModelFamily[]
  provider: ModelOptionProvider
}

export function ModelMenuPanel({ gateway, onSelectModel, profile = 'default', requestGateway }: ModelMenuPanelProps) {
  const { t } = useI18n()
  const copy = t.shell.modelMenu
  const closeMenu = useContext(ModelMenuCloseContext)
  const [search, setSearch] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const queryClient = useQueryClient()
  // Bind to THIS surface's SessionView (primary or tile) so each pane's menu
  // shows/switches its own model — not the primary-only globals.
  const view = useSessionView()
  const activeSessionId = useStore(view.$runtimeId)
  const currentFastMode = useStore(view.$fast)
  const currentModel = useStore(view.$model)
  const currentProvider = useStore(view.$provider)
  const currentReasoningEffort = useStore(view.$reasoningEffort)
  const modelPresets = useStore($modelPresets)
  const defaultEffort = useStore($defaultReasoningEffort) || DEFAULT_REASONING_EFFORT
  const visibleModels = useStore($visibleModels)
  const collapsedProviders = useStore($collapsedProviders)

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, activeSessionId),
    // Gateway-first even with no session yet: a connected (possibly remote)
    // gateway owns the model catalog, including virtual providers like `moa`
    // that the local REST fallback can't know about (#53817).
    queryFn: (): Promise<ModelOptionsResponse> => requestModelOptions({ gateway, sessionId: activeSessionId })
  })

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
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
  // Always stamp sessionId from this surface so a tile switch never hits the
  // primary (busy) session by accident.
  const switchTo = (model: string, provider: string) =>
    onSelectModel({ model, provider, sessionId: activeSessionId || null })

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
      const queryKey = modelOptionsQueryKey(profile, activeSessionId)

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
        effort: (caps?.reasoning ?? true) ? (preset.effort ?? defaultEffort) : undefined,
        fast: (caps?.fast ?? false) ? (preset.fast ?? false) : undefined
      },
      {
        failMessage: t.shell.modelOptions.updateFailed,
        primary: view.kind === 'primary',
        request: requestGateway,
        sessionId: activeSessionId
      }
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

  const q = normalize(search)

  // Presets are searchable rows like everything else — an unfiltered preset
  // sitting under zero model matches would otherwise become the "first match"
  // Enter commits.
  const shownMoaPresets = useMemo(
    () => (q ? moaPresets.filter(preset => `moa ${preset}`.toLowerCase().includes(q)) : moaPresets),
    [moaPresets, q]
  )

  // ── Keyboard selection (cmdk semantics on a Radix menu) ───────────────────
  // One flat list mirroring EXACTLY what's rendered (collapse, filter, presets),
  // so the selection can never sit on a hidden row. The selected index is
  // derived — current model with no query (Enter = close), first match while
  // typing — with an arrow-key override that resets on every keystroke. Focus
  // stays in the search input throughout: ⌘⇧M → type → ↑/↓ → Enter.
  type KbRow =
    | { family: ModelFamily; key: string; kind: 'family'; provider: ModelOptionProvider }
    | { key: string; kind: 'moa'; preset: string }

  const kbRows = useMemo<KbRow[]>(
    () => [
      ...groups.flatMap(group =>
        collapsedProviders.includes(group.provider.slug) && !search
          ? []
          : group.families.map((family): KbRow => ({
              family,
              key: `${group.provider.slug}:${family.id}`,
              kind: 'family',
              provider: group.provider
            }))
      ),
      ...shownMoaPresets.map((preset): KbRow => ({ key: `moa:${preset}`, kind: 'moa', preset }))
    ],
    [groups, collapsedProviders, search, shownMoaPresets]
  )

  const [kbOverride, setKbOverride] = useState<null | number>(null)
  // A parked cursor is not a cursor in use: until the mouse actually moves,
  // hover can't take rows out from under the keyboard (rows re-flow beneath it
  // as the filter narrows). One real movement hands hover back.
  const pointerQuiet = usePointerQuiet()

  const currentKey = optionsProvider === 'moa' ? `moa:${optionsModel}` : `${optionsProvider}:${optionsModel}`

  const autoIndex = q
    ? kbRows.length > 0
      ? 0
      : -1
    : kbRows.findIndex(row => row.key === currentKey || (row.kind === 'family' && row.family.fastId === optionsModel))

  const kbIndex = kbOverride !== null && kbOverride < kbRows.length ? kbOverride : autoIndex
  const kbActiveKey = kbIndex >= 0 ? kbRows[kbIndex].key : null

  const stepKb = (delta: -1 | 1) => {
    if (kbRows.length === 0) {
      return
    }

    const from = kbIndex >= 0 ? kbIndex : delta === 1 ? -1 : 0

    setKbOverride((from + delta + kbRows.length) % kbRows.length)
  }

  const commitKbRow = () => {
    const row = kbIndex >= 0 ? kbRows[kbIndex] : undefined

    if (!row) {
      return
    }

    if (row.kind === 'moa') {
      void selectMoaPreset(row.preset)

      return
    }

    if (row.key !== currentKey && row.family.fastId !== optionsModel) {
      void selectFamily(row.family, row.provider)
    }

    closeMenu()
  }

  // Keep the selected row in view while arrowing through the scrollable list.
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.querySelector('[data-kb-active]')?.scrollIntoView({ block: 'nearest' })
  }, [kbActiveKey])

  // The keyboard-selected row, styled + tagged for scrollIntoView. Pointer
  // suppression is NOT here — it belongs on the containers (below), so one
  // class covers every row inside them.
  const kbRowProps = (key: string) => {
    const active = kbActiveKey === key

    return {
      className: cn(dropdownMenuRow, active && 'bg-(--ui-control-active-background) text-foreground'),
      ...(active ? { 'data-kb-active': '' } : {})
    }
  }

  // Rows are hover-selectable, so they go inert with the pointer (usePointerQuiet).
  const quietRows = pointerQuiet && 'pointer-events-none'

  return (
    <>
      <DropdownMenuSearch
        aria-label={copy.search}
        onKeyDown={event => {
          // Claim arrows and Enter from Radix so DOM focus stays in the input
          // and Enter commits the highlighted row without a DownArrow first
          // (VS Code's checked-or-first pattern).
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault()
            event.stopPropagation()
            stepKb(event.key === 'ArrowDown' ? 1 : -1)
          } else if (event.key === 'Enter') {
            event.preventDefault()
            event.stopPropagation()
            commitKbRow()
          }
        }}
        onValueChange={value => {
          setSearch(value)
          setKbOverride(null)
        }}
        placeholder={copy.search}
        value={search}
      />

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
        <div className={cn('max-h-[max(150px,30dvh)] overflow-y-auto py-0.5', quietRows)} ref={listRef}>
          {groups.map(group => {
            const slug = group.provider.slug

            // Collapsed when the user stored it (and not while searching, which
            // spans every model regardless of collapse state).
            const collapsed = collapsedProviders.includes(slug) && !search

            return (
              <DropdownMenuGroup className="py-0.5" key={slug}>
                <DropdownMenuItem
                  className="group/label flex w-full items-center gap-1 px-2 pb-0.5 pt-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary) cursor-pointer !bg-transparent focus:!bg-transparent"
                  onSelect={event => {
                    event.preventDefault()
                    toggleCollapsedProvider(slug)
                  }}
                  textValue=""
                >
                  <span className="truncate">
                    <HighlightMatches query={search} text={group.provider.name} />
                  </span>
                  <DisclosureCaret
                    className="shrink-0 text-(--ui-text-tertiary) opacity-0 transition group-hover/label:opacity-100"
                    open={!collapsed}
                    size="0.625rem"
                  />
                </DropdownMenuItem>
                {!collapsed &&
                  group.families.map(family => {
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
                      (caps?.reasoning ?? true) ? reasoningEffortLabel(effEffort || defaultEffort) : null
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
                          hideChevron
                          onClick={activate}
                          onKeyDown={event => {
                            if (event.key === 'Enter' || event.key === ' ') {
                              activate()
                            }
                          }}
                          {...kbRowProps(`${group.provider.slug}:${family.id}`)}
                        >
                          <span className="min-w-0 flex-1 truncate">
                            <HighlightMatches query={search} text={name} />
                            {meta ? <span className="text-(--ui-text-tertiary)"> {meta}</span> : null}
                          </span>
                          {isCurrent ? (
                            <Codicon className="ml-auto text-foreground" name="check" size="0.75rem" />
                          ) : null}
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
            )
          })}
        </div>
      )}

      <DropdownMenuSeparator className="mx-0" />

      {shownMoaPresets.length > 0 ? (
        <div className={cn(quietRows)}>
          <DropdownMenuLabel className={dropdownMenuSectionLabel}>MoA presets</DropdownMenuLabel>
          {shownMoaPresets.map(preset => {
            const isCurrentMoa = optionsProvider === 'moa' && optionsModel === preset

            return (
              <DropdownMenuItem
                key={`moa:${preset}`}
                onSelect={event => {
                  event.preventDefault()
                  void selectMoaPreset(preset)
                }}
                {...kbRowProps(`moa:${preset}`)}
              >
                <span className="min-w-0 flex-1 truncate">
                  MoA: <HighlightMatches query={search} text={preset} />
                </span>
                {isCurrentMoa ? <Codicon className="ml-auto text-foreground" name="check" size="0.75rem" /> : null}
              </DropdownMenuItem>
            )
          })}
          <DropdownMenuSeparator className="mx-0" />
        </div>
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
    // a model can't shuffle the list. While SEARCHING, the pin is skipped: a
    // query means "show me matches", and a pinned non-match sitting above them
    // reads like the top result (type "grok", see the current Fable first).
    const activeId =
      !q && provider.slug === current.provider && current.model
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

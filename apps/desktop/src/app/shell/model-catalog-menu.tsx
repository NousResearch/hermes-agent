import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared'
import { DEFAULT_REASONING_EFFORT } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import {
  createContext,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
  type SyntheticEvent,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

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
import { getLocalModelsStatus } from '@/hermes'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { catalogProviderMatches, modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { reasoningEffortLabel } from '@/lib/reasoning-effort'
import { foldIncludes, normalize } from '@/lib/text'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs, runningModelDownloads, watchLocalRuntimeJobs } from '@/store/local-runtime-jobs'
import { $pinnedModelKeys, toggleModelPinned } from '@/store/model-pinned'
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
import type { LocalModelLoadProgress } from '@/types/hermes'

import { type FastControl, ModelEditSubmenu, resolveFastControl } from './model-edit-submenu'

// Lets the host dropdown (model-pill, a kanban field trigger, …) hand the panel
// a way to dismiss itself so clicking a model row commits + closes, while the
// hover-revealed edit submenu (reasoning/fast) stays open to play with (its
// items preventDefault on select).
export const ModelMenuCloseContext = createContext<() => void>(() => {})

/** One model choice, everything a caller needs to act on a selection.
 *  `effort` is '' for "inherit the default" and 'none' for thinking off. */
export interface ModelChoice {
  effort: string
  fast: boolean
  model: string
  provider: string
}

/**
 * What a surface DOES with the catalog. The menu renders and navigates; the
 * controller owns meaning — the composer writes through to a live session,
 * the kanban override just holds a value in dialog state.
 *
 * `presetFor` supplies the remembered settings shown on a non-active row.
 * Returning `{}` is fine — the row then shows Hermes' defaults.
 */
export interface ModelMenuController {
  /** Restore a model's remembered settings after it is selected. Separate from
   *  `setOptions` because it is one atomic "apply this model's preset" write,
   *  not a user editing one control — surfaces that write through to a session
   *  need to batch it. Values are already capability-gated by the menu. */
  applyPreset: (preset: { effort?: string; fast?: boolean }, row: { model: string; provider: string }) => void
  current: ModelChoice
  presetFor: (provider: string, model: string) => { effort?: string; fast?: boolean }
  /** Commit a model row. Return false to abort (a failed session switch). */
  select: (model: string, provider: string) => Promise<boolean | void> | void
  /** Edit ONE option on a row. `isActive` says whether it's the current model. */
  setOptions: (
    patch: { effort?: string; fast?: boolean },
    row: { isActive: boolean; model: string; provider: string }
  ) => void
}

interface ModelCatalogMenuProps {
  controller: ModelMenuController
  /** Rows appended under the catalog (Refresh Models, Edit Models, …). */
  footer?: ReactNode
  gateway?: HermesGateway
  /** Owner-routed RPC for catalog reads. Preferred over `gateway.request` so
   *  a tile's menu queries the session owner's backend, not chrome's. */
  request?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  /** Render the virtual `moa` provider's presets as a selectable section.
   *  Off for override surfaces, where a MoA preset isn't a worker model. */
  includeMoa?: boolean
  /** Registry source owning this catalog. Profile/session names are not unique
   * across sources, so this participates in the React Query cache key. */
  ownerConnectionId?: string
  profile?: string
  /** Session whose catalog to fetch. A live session's catalog can differ from
   *  the profile-global one, and the app invalidates the SESSION-scoped query
   *  key on model changes — a surface bound to a session must pass it or its
   *  menu goes stale. Detached surfaces (per-task overrides) omit it. */
  sessionId?: null | string
}

interface ProviderGroup {
  families: ModelFamily[]
  provider: ModelOptionProvider
}

/** A pinned model resolved against the live catalog: which family, under
 *  which provider. Cross-provider by design — pins live in one flat list,
 *  not scoped to a provider group. */
interface PinnedRow {
  family: ModelFamily
  provider: ModelOptionProvider
}

/**
 * THE model catalog menu: searchable, provider-grouped, `-fast` families
 * collapsed to one row, per-row hover submenu for thinking/effort/fast, full
 * keyboard selection. Shared verbatim by the composer's model pill and by
 * plugin surfaces that pick a model without a session behind it — so the two
 * can never drift apart.
 */
export function ModelCatalogMenu({
  controller,
  footer,
  gateway,
  includeMoa = false,
  ownerConnectionId,
  profile = 'default',
  request,
  sessionId = null
}: ModelCatalogMenuProps) {
  const { t } = useI18n()
  const copy = t.shell.modelMenu
  const copyPicker = t.modelPicker
  const closeMenu = useContext(ModelMenuCloseContext)
  const [search, setSearch] = useState('')
  const collapsedProviders = useStoreCollapsed()
  const defaultEffort = useDefaultEffort()
  // Which models the user curated in Edit Models. Read HERE rather than taken
  // as a prop: it's one global preference, so every surface that shows a
  // catalog must show the same shortlist. A per-caller opt-in is how the board
  // and the composer would end up disagreeing about what "my models" means.
  const visibleModels = useStore($visibleModels)

  // Which models the user pinned to the top, cross-provider. Same rationale
  // as visibleModels: one global preference, read here so every surface
  // (composer pill, kanban override, …) agrees on the pinned set.
  const pinnedKeys = useStore($pinnedModelKeys)
  const pinnedKeySet = useMemo(() => new Set(pinnedKeys), [pinnedKeys])

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId, ownerConnectionId),
    // Gateway-first even with no session: a connected (possibly remote)
    // gateway owns the model catalog, including virtual providers the local
    // REST fallback can't know about (#53817).
    queryFn: (): Promise<ModelOptionsResult> => requestModelOptions({ gateway, profile, request, sessionId })
  })

  const loading = modelOptions.isPending && !modelOptions.data

  // Every local-models read in this menu sits behind the --local launch
  // flag: no status polling, no download rows, and the llamacpp provider
  // group hides even when models are staged (the flag is strict).
  const localModelsEnabled = $localModelsEnabled.get()

  // Live load state for the managed local server: which model is loading
  // into memory right now, with a REAL percent (per-tensor callback relayed
  // over the router's SSE stream). Polled only while this menu is mounted
  // (it unmounts on close); errors read as "nothing loading" — remote-only
  // installs have no local-models routes.
  const localStatus = useQuery({
    queryKey: ['local-models-loading', profile],
    queryFn: () => getLocalModelsStatus(),
    enabled: localModelsEnabled,
    refetchInterval: 2_000,
    retry: false
  })

  const loadingModels: Record<string, LocalModelLoadProgress> = localStatus.data?.loading ?? {}

  // Models on their way into the local library (downloads + quickstart runs
  // still fetching bytes) — rendered as disabled progress rows so the user
  // sees the model coming instead of wondering where it went. The jobs store
  // republishes every ~700ms with fresh byte counts while anything runs; a
  // whole-store subscription here would re-render the entire menu per tick
  // (breaking open submenus and focus — the #72163 class). Subscribe to a
  // STABLE identity projection instead: it changes only when a download
  // starts or ends. Each row selects its own percent scalar.
  const downloadsKey = useStoreSelector($localRuntimeJobs, jobs =>
    localModelsEnabled
      ? runningModelDownloads(jobs)
          .map(job => `${job.job_id}\u0000${job.target}`)
          .join('\u0001')
      : ''
  )

  const downloads = useMemo(
    () =>
      downloadsKey === ''
        ? []
        : downloadsKey.split('\u0001').map(pair => {
            const [jobId, target] = pair.split('\u0000')

            return { jobId, target }
          }),
    [downloadsKey]
  )

  useEffect(() => {
    if (localModelsEnabled) {
      watchLocalRuntimeJobs()
    }
  }, [localModelsEnabled])

  // A finished download turns into a real selectable model: refetch the
  // catalog so the placeholder row is replaced while the menu is open.
  const refetchOptions = modelOptions.refetch

  useEffect(() => {
    let prevActive = runningModelDownloads($localRuntimeJobs.get()).length > 0

    return $localRuntimeJobs.listen(next => {
      const active = runningModelDownloads(next).length > 0

      if (prevActive && !active) {
        void refetchOptions()
      }

      prevActive = active
    })
  }, [refetchOptions])

  const error = modelOptions.error
    ? modelOptions.error instanceof Error
      ? modelOptions.error.message
      : String(modelOptions.error)
    : null

  const providers = modelOptions.data?.providers

  // The catalog carries MoA presets as a virtual `moa` provider row. Keep it
  // out of the main groups so presets never show up twice.
  const moaPresets = useMemo(
    () => (includeMoa ? (providers?.find(p => p.slug.toLowerCase() === 'moa')?.models ?? []) : []),
    [providers, includeMoa]
  )

  const pickerProviders = useMemo(
    () =>
      providers?.filter(
        provider =>
          provider.slug.toLowerCase() !== 'moa' &&
          // Strict --local gate: staged local models exist on disk, but
          // without the flag the GUI doesn't offer them.
          (localModelsEnabled || provider.slug !== LOCAL_PROVIDER_SLUG)
      ) ?? [],
    [providers, localModelsEnabled]
  )

  const current = controller.current

  const q = normalize(search)

  // In-flight downloads render inside the Local provider group when it
  // exists, else as their own trailing 'Local' group (first download —
  // nothing staged yet, so the catalog has no local provider row).
  const shownDownloads = q ? downloads.filter(job => foldIncludes(job.target || '', q)) : downloads
  const hasLocalGroup = pickerProviders.some(provider => provider.slug === LOCAL_PROVIDER_SLUG)

  // Resolve visibility HERE, against the catalog we actually fetched: an empty
  // provider list would otherwise resolve to an empty key set that reads as
  // "user hid everything" and blanks the menu on first open.
  const shownKeys = useMemo(
    () => effectiveVisibleKeys(visibleModels, pickerProviders),
    [visibleModels, pickerProviders]
  )

  // Pins are resolved against the FULL catalog (not `shownKeys`): a pin is an
  // explicit "show me this" signal that outranks an Edit Models hide, exactly
  // like the active model already does below in `groupModels`. Search still
  // applies — a pin that doesn't match the query is hidden like anything else.
  const pinnedRows = useMemo(() => resolvePinnedRows(pickerProviders, pinnedKeys, search), [pickerProviders, pinnedKeys, search])

  // Once a pin is shown in the Pinned section, drop it from its provider
  // group so it never renders twice.
  const pinnedFamilyKeys = useMemo(
    () => new Set(pinnedRows.map(row => modelVisibilityKey(row.provider.slug, row.family.id))),
    [pinnedRows]
  )

  const groups = useMemo(
    () =>
      groupModels(
        pickerProviders,
        search,
        { model: current.model, provider: current.provider },
        shownKeys,
        pinnedFamilyKeys
      ),
    [pickerProviders, search, current.model, current.provider, shownKeys, pinnedFamilyKeys]
  )

  // Presets are searchable rows like everything else — an unfiltered preset
  // sitting under zero model matches would otherwise become the "first match"
  // Enter commits.
  const shownMoaPresets = useMemo(
    () => (q ? moaPresets.filter(preset => foldIncludes(`moa ${preset}`, q)) : moaPresets),
    [moaPresets, q]
  )

  const selectFamily = async (family: ModelFamily, provider: ModelOptionProvider) => {
    const caps = provider.capabilities?.[family.id]
    const preset = controller.presetFor(provider.slug, family.id)

    // Variant-fast models (no speed param) express "fast" as a separate `-fast`
    // id, so honor the remembered preset by selecting that sibling. Param-fast
    // is applied through setOptions below instead.
    const variantFast = !(caps?.fast ?? false) && !!family.fastId
    const targetId = variantFast && preset.fast === true ? family.fastId! : family.id

    if ((await controller.select(targetId, provider.slug)) === false) {
      return
    }

    controller.applyPreset(
      {
        effort: (caps?.reasoning ?? true) ? (preset.effort ?? defaultEffort) : undefined,
        fast: (caps?.fast ?? false) ? (preset.fast ?? false) : undefined
      },
      { model: family.id, provider: provider.slug }
    )
  }

  const selectMoaPreset = async (preset: string) => {
    if ((await controller.select(preset, 'moa')) === false) {
      return
    }

    closeMenu()
  }

  // ── Keyboard selection (cmdk semantics on a Radix menu) ───────────────────
  // One flat list mirroring EXACTLY what's rendered (pinned, collapse, filter,
  // presets), so the selection can never sit on a hidden row. Pinned rows come
  // first — that's also their visual position.
  type KbRow =
    | { family: ModelFamily; key: string; kind: 'family'; provider: ModelOptionProvider }
    | { key: string; kind: 'moa'; preset: string }

  const kbRows = useMemo<KbRow[]>(
    () => [
      ...pinnedRows.map(
        (row): KbRow => ({
          family: row.family,
          key: pinnedRowKey(row.provider.slug, row.family.id),
          kind: 'family',
          provider: row.provider
        })
      ),
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
    [pinnedRows, groups, collapsedProviders, search, shownMoaPresets]
  )

  const [kbOverride, setKbOverride] = useState<null | number>(null)
  // A parked cursor is not a cursor in use: until the mouse actually moves,
  // hover can't take rows out from under the keyboard.
  const pointerQuiet = usePointerQuiet()

  const rowIsCurrent = (row: KbRow) =>
    row.kind === 'moa'
      ? current.provider === 'moa' && row.preset === current.model
      : catalogProviderMatches(row.provider, current.provider) &&
        (row.family.id === current.model || row.family.fastId === current.model)

  const autoIndex = q ? (kbRows.length > 0 ? 0 : -1) : kbRows.findIndex(row => rowIsCurrent(row))

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

    if (!rowIsCurrent(row)) {
      void selectFamily(row.family, row.provider)
    }

    closeMenu()
  }

  // Keep the selected row in view while arrowing through the scrollable list.
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.querySelector('[data-kb-active]')?.scrollIntoView({ block: 'nearest' })
  }, [kbActiveKey])

  const kbRowProps = (key: string) => {
    const active = kbActiveKey === key

    return {
      className: cn(dropdownMenuRow, active && 'bg-(--ui-control-active-background) text-foreground'),
      ...(active ? { 'data-kb-active': '' } : {})
    }
  }

  // Whether a row (by its keyboard-list key) is the current keyboard cursor —
  // read separately from `kbRowProps` so callers that spread the DOM props
  // onto an element never leak this boolean as an unknown attribute.
  const isKbActive = (key: string) => kbActiveKey === key

  // Rows are hover-selectable, so they go inert with the pointer.
  const quietRows = pointerQuiet && 'pointer-events-none'

  // Shared row renderer: pinned rows and per-provider rows are the exact same
  // row (checkmark, fast/effort meta, load progress, pin toggle, hover
  // submenu) — factored out so the Pinned section and the provider groups can
  // never drift apart.
  const renderFamilyRow = (family: ModelFamily, provider: ModelOptionProvider, key: string) => {
    // The active id may be the base or its -fast sibling; either way this one
    // family row represents both.
    const activeId =
      catalogProviderMatches(provider, current.provider) && (current.model === family.id || current.model === family.fastId)
        ? current.model
        : null

    const isCurrent = activeId !== null
    const name = modelDisplayParts(family.id).name
    const caps = provider.capabilities?.[family.id]

    // Managed local model loading into memory right now: real load percent,
    // keyed by exact model id (remote providers never collide with GGUF stems).
    const loadProgress = loadingModels[family.id] ?? (family.fastId ? loadingModels[family.fastId] : undefined)

    // Effective settings for this row: the live choice when it's the active
    // model, otherwise its remembered preset. Row label AND submenu read from
    // these so they never disagree.
    const preset = controller.presetFor(provider.slug, family.id)
    const effEffort = isCurrent ? current.effort : (preset.effort ?? '')
    const effFast = isCurrent ? current.fast : (preset.fast ?? false)

    const fastControl: FastControl = resolveFastControl(
      activeId ?? family.id,
      provider.models ?? [],
      caps?.fast ?? false,
      effFast
    )

    const meta = [
      fastControl.kind !== 'none' && fastControl.on ? copy.fast : null,
      (caps?.reasoning ?? true) ? reasoningEffortLabel(effEffort || defaultEffort) : null
    ]
      .filter(Boolean)
      .join(' ')

    // Clicking the row commits the model and closes; the edit submenu
    // (reasoning/fast) is reached by HOVER, so you can tweak those without
    // the click dismissing everything.
    const activate = () => {
      if (!isCurrent) {
        void selectFamily(family, provider)
      }

      closeMenu()
    }

    const pinKey = modelVisibilityKey(provider.slug, family.id)
    const pinned = pinnedKeySet.has(pinKey)
    const rowProps = kbRowProps(key)
    const kbActive = isKbActive(key)

    // The pin toggle sits INSIDE a row that also commits on click — stop the
    // click (and the pointerdown Radix uses for its own press handling) from
    // reaching the row's own handler, or pinning would also select the model.
    const togglePin = (event: ReactMouseEvent | SyntheticEvent) => {
      event.preventDefault()
      event.stopPropagation()
      toggleModelPinned(pinKey)
    }

    return (
      <DropdownMenuSub key={key}>
        <DropdownMenuSubTrigger
          hideChevron
          onClick={activate}
          onKeyDown={event => {
            if (event.key === 'Enter' || event.key === ' ') {
              activate()
            }
          }}
          {...rowProps}
          className={cn(rowProps.className, 'group/row relative')}
        >
          <button
            aria-label={pinned ? copy.unpin : copy.pin}
            aria-pressed={pinned}
            className={cn(
              'shrink-0 rounded-sm p-0.5 text-(--ui-text-tertiary) transition-opacity hover:text-foreground',
              pinned || kbActive ? 'opacity-100' : 'opacity-0 group-hover/row:opacity-100'
            )}
            onClick={togglePin}
            onKeyDown={event => {
              if (event.key === 'Enter' || event.key === ' ') {
                togglePin(event)
              }
            }}
            onPointerDown={event => event.stopPropagation()}
            tabIndex={-1}
            type="button"
          >
            <Codicon className={pinned ? 'text-foreground' : undefined} name={pinned ? 'pinned' : 'pin'} size="0.75rem" />
          </button>
          <span className="min-w-0 flex-1 truncate">
            <HighlightMatches foldSeparators query={search} text={name} />
            {meta ? <span className="text-(--ui-text-tertiary)"> {meta}</span> : null}
          </span>
          {loadProgress ? (
            <span className="ml-auto flex shrink-0 items-center gap-1.5" title={copyPicker.loadingIntoMemory}>
              <span className="h-1 w-14 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
                <span
                  className="block h-full rounded-full bg-primary transition-[width] duration-500"
                  style={{ width: `${Math.max(2, loadProgress.percent)}%` }}
                />
              </span>
              <span className="text-[0.62rem] tabular-nums text-(--ui-text-tertiary)">{loadProgress.percent}%</span>
            </span>
          ) : null}
          {isCurrent ? (
            <Codicon
              className={cn('text-foreground', loadProgress ? 'ml-1' : 'ml-auto')}
              name="check"
              size="0.75rem"
            />
          ) : null}
        </DropdownMenuSubTrigger>
        <ModelEditSubmenu
          canDisableReasoning={caps?.can_disable_reasoning ?? undefined}
          defaultEffort={defaultEffort}
          effort={effEffort}
          fastControl={fastControl}
          isActive={isCurrent}
          model={family.id}
          onSelectModel={nextModel => controller.select(nextModel, provider.slug)}
          onSetOptions={patch =>
            controller.setOptions(patch, {
              isActive: isCurrent,
              model: family.id,
              provider: provider.slug
            })
          }
          provider={provider.slug}
          reasoning={caps?.reasoning ?? true}
        />
      </DropdownMenuSub>
    )
  }

  return (
    <>
      <DropdownMenuSearch
        aria-label={copy.search}
        onKeyDown={event => {
          // Claim arrows and Enter from Radix so DOM focus stays in the input
          // and Enter commits the highlighted row without a DownArrow first.
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault()
            event.stopPropagation()
            stepKb(event.key === 'ArrowDown' ? 1 : -1)
          } else if (isSubmitEnter(event)) {
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
      ) : groups.length === 0 && moaPresets.length === 0 && shownDownloads.length === 0 && pinnedRows.length === 0 ? (
        <DropdownMenuItem className={dropdownMenuRow} disabled>
          {copy.noModels}
        </DropdownMenuItem>
      ) : (
        <div className={cn('max-h-[max(150px,30dvh)] overflow-y-auto py-0.5', quietRows)} ref={listRef}>
          {pinnedRows.length > 0 && (
            <DropdownMenuGroup className="py-0.5" key="pinned">
              <DropdownMenuLabel className="px-2 pb-0.5 pt-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
                {copy.pinned}
              </DropdownMenuLabel>
              {pinnedRows.map(row =>
                renderFamilyRow(row.family, row.provider, pinnedRowKey(row.provider.slug, row.family.id))
              )}
            </DropdownMenuGroup>
          )}
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
                    <HighlightMatches foldSeparators query={search} text={group.provider.name} />
                  </span>
                  <DisclosureCaret
                    className="shrink-0 text-(--ui-text-tertiary) opacity-0 transition group-hover/label:opacity-100"
                    open={!collapsed}
                    size="0.625rem"
                  />
                </DropdownMenuItem>
                {!collapsed && group.families.map(family => renderFamilyRow(family, group.provider, `${slug}:${family.id}`))}
                {!collapsed &&
                  slug === LOCAL_PROVIDER_SLUG &&
                  shownDownloads.map(job => (
                    <DownloadingModelRow jobId={job.jobId} key={job.jobId} target={job.target} />
                  ))}
              </DropdownMenuGroup>
            )
          })}
          {!hasLocalGroup && shownDownloads.length > 0 && (
            <DropdownMenuGroup className="py-0.5" key="local-downloads">
              <DropdownMenuLabel className="px-2 pb-0.5 pt-0.5 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
                {copyPicker.localDownloadsHeading}
              </DropdownMenuLabel>
              {shownDownloads.map(job => (
                <DownloadingModelRow jobId={job.jobId} key={job.jobId} target={job.target} />
              ))}
            </DropdownMenuGroup>
          )}
        </div>
      )}

      {shownMoaPresets.length > 0 ? (
        <div className={cn(quietRows)}>
          <DropdownMenuSeparator className="mx-0" />
          <DropdownMenuLabel className={dropdownMenuSectionLabel}>MoA presets</DropdownMenuLabel>
          {shownMoaPresets.map(preset => {
            const isCurrentMoa = current.provider === 'moa' && current.model === preset

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
                  MoA: <HighlightMatches foldSeparators query={search} text={preset} />
                </span>
                {isCurrentMoa ? <Codicon className="ml-auto text-foreground" name="check" size="0.75rem" /> : null}
              </DropdownMenuItem>
            )
          })}
        </div>
      ) : null}

      {/* Curation belongs to the catalog, not to one host: wherever you can
          pick a model you can say which models you want, and the shortlist is
          the same everywhere because it's one stored preference. It shares the
          host footer's group rather than opening a second one, so a host that
          contributes rows (the composer's Refresh Models) keeps the single
          trailing block it has always rendered. */}
      <DropdownMenuSeparator className="mx-0" />
      {footer}
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

/** Re-exported so callers building a footer row match the catalog's rows. */
export { dropdownMenuRow }

// The backend's provider row for staged local models (inventory.py's
// _local_runtime_row). Downloads-in-flight attach to this group.
const LOCAL_PROVIDER_SLUG = 'llamacpp'

// A model still downloading: visible so the user knows it's coming (and
// where it will land), disabled so it can't be selected early, with the
// same byte progress the Local Models pane shows. Percent is selected HERE,
// per row, so the 700ms byte ticks repaint this leaf only — the menu tree
// above subscribes to download identity, not progress.
function DownloadingModelRow({ jobId, target }: { jobId: string; target: string }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  const percent = useStoreSelector($localRuntimeJobs, jobs => jobs.find(job => job.job_id === jobId)?.percent ?? null)

  return (
    <DropdownMenuItem
      className={cn(dropdownMenuRow, 'opacity-60')}
      disabled
      onSelect={event => event.preventDefault()}
      textValue=""
    >
      <span className="min-w-0 flex-1 truncate">{target}</span>
      <span className="ml-auto flex shrink-0 items-center gap-1.5" title={copy.downloading}>
        <span className="h-1 w-14 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
          <span
            className="block h-full rounded-full bg-primary transition-[width] duration-500"
            style={{ width: `${Math.max(2, percent ?? 0)}%` }}
          />
        </span>
        <span className="text-[0.62rem] tabular-nums text-(--ui-text-tertiary)">
          {typeof percent === 'number' ? `${percent}%` : copy.downloading}
        </span>
      </span>
    </DropdownMenuItem>
  )
}

// Stable key for a pinned row in the keyboard flat list and React `key`s —
// prefixed so it can never collide with the same model's key inside its
// (now-excluded) provider group.
function pinnedRowKey(providerSlug: string, familyId: string): string {
  return `pinned:${providerSlug}:${familyId}`
}

// Shared by `resolvePinnedRows` and `groupModels` so a pin and its provider
// group can never disagree about what a search query matches.
function familyMatchesQuery(family: ModelFamily, provider: ModelOptionProvider, q: string): boolean {
  return foldIncludes(
    `${family.id} ${family.fastId ?? ''} ${provider.name} ${provider.slug} ${displayModelName(family.id)}`,
    q
  )
}

// Resolve stored pin keys (`provider::model`, oldest pin first — same
// insertion-order contract as session pins in `store/layout.ts`) against the
// live catalog. A pin outlives an Edit Models hide (resolved against every
// family, not `shownKeys`); a stale pin (model retired upstream, or a
// provider no longer configured) silently drops rather than erroring.
function resolvePinnedRows(providers: ModelOptionProvider[], pinnedKeys: string[], search: string): PinnedRow[] {
  const q = normalize(search)
  const rows: PinnedRow[] = []

  for (const key of pinnedKeys) {
    // Keys are `${provider}::${model}` (model-visibility.ts's format) — model
    // ids may contain a single colon, so split on the FIRST `::` only.
    const separator = key.indexOf('::')

    if (separator < 0) {
      continue
    }

    const providerSlug = key.slice(0, separator)
    const modelId = key.slice(separator + 2)
    const provider = providers.find(p => p.slug === providerSlug)

    if (!provider) {
      continue
    }

    const family = collapseModelFamilies(provider.models ?? []).find(f => f.id === modelId || f.fastId === modelId)

    if (!family) {
      continue
    }

    if (q && !familyMatchesQuery(family, provider, q)) {
      continue
    }

    rows.push({ family, provider })
  }

  return rows
}

// Collapsed we show the user's chosen models (or the curated default); typing
// spans every available model so anything is reachable past the cut. A search
// is itself a narrowing action, so we do NOT cap per-provider matches.
// `excludeFamilyKeys` drops models already shown in the Pinned section so a
// pin never renders twice.
function groupModels(
  providers: ModelOptionProvider[],
  search: string,
  current: { model: string; provider: string },
  visible: Set<string> | null,
  excludeFamilyKeys: Set<string>
): ProviderGroup[] {
  const q = normalize(search)
  const groups: ProviderGroup[] = []

  for (const provider of providers) {
    const allFamilies = collapseModelFamilies(provider.models ?? [])

    if (allFamilies.length === 0) {
      continue
    }

    let shown: Set<string>

    if (q) {
      // Search spans every family, regardless of visibility.
      shown = new Set(allFamilies.filter(family => familyMatchesQuery(family, provider, q)).map(family => family.id))
    } else if (visible) {
      // User has customized which models show — honor their selection exactly.
      shown = new Set(
        allFamilies.filter(family => visible.has(modelVisibilityKey(provider.slug, family.id))).map(family => family.id)
      )
    } else {
      shown = new Set(allFamilies.slice(0, DEFAULT_VISIBLE_PER_PROVIDER).map(family => family.id))
    }

    // Always include the active model — but keep every row in the provider's
    // stable curated order, so selecting a model can't shuffle the list. While
    // SEARCHING the pin is skipped: a query means "show me matches".
    const activeId =
      !q && catalogProviderMatches(provider, current.provider) && current.model
        ? allFamilies.find(family => family.id === current.model || family.fastId === current.model)?.id
        : undefined

    const families = allFamilies.filter(
      family =>
        (shown.has(family.id) || family.id === activeId) &&
        !excludeFamilyKeys.has(modelVisibilityKey(provider.slug, family.id))
    )

    if (families.length > 0) {
      groups.push({ families, provider })
    }
  }

  // Stable, logical group order: alphabetical by provider name. (The backend
  // floats the current provider first, which would reshuffle on every switch.)
  groups.sort((a, b) => a.provider.name.localeCompare(b.provider.name))

  return groups
}

// Small hooks kept at the bottom so the component reads top-down.
function useStoreCollapsed(): string[] {
  return useStore($collapsedProviders)
}

function useDefaultEffort(): string {
  return useStore($defaultReasoningEffort) || DEFAULT_REASONING_EFFORT
}

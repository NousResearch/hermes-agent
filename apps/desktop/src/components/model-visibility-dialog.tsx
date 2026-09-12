import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { memo, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { HoverScroll } from '@/components/ui/marquee'
import { Switch } from '@/components/ui/switch'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { Search } from '@/lib/icons'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { compareModelIds, displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { normalize, searchFold } from '@/lib/text'
import { useDebouncedValue } from '@/lib/use-debounced-value'
import {
  $visibleModels,
  collapseModelFamilies,
  effectiveVisibleKeys,
  modelVisibilityKey,
  setProviderVisibility,
  setVisibleModels,
  toggleModelVisibility
} from '@/store/model-visibility'
import { $collapsedProviders, toggleCollapsedProvider } from '@/store/provider-collapse'
import type { ModelOptionProvider, ModelOptionsResponse } from '@/types/hermes'

/** Render cap per provider — filtering thousands of ids is cheap, mounting
 *  thousands of toggle rows is not. The expander below lifts it per provider. */
const DIALOG_RENDER_CAP = 200

/** One toggle row, memoized so a switch flips one row instead of re-rendering
 *  hundreds (the per-row toggle closure is intentionally excluded from the
 *  compare — it reads fresh stores itself). No pills: the id line below
 *  already carries the upstream verbatim. */
const VisibilityRow = memo(
  function VisibilityRow({
    checked,
    familyId,
    onToggle,
    provider,
    search,
    terms
  }: {
    checked: boolean
    familyId: string
    onToggle: () => void
    provider: ModelOptionProvider
    search: string
    terms: string | string[]
  }) {
    const [go, setGo] = useState(false)
    const { name } = modelDisplayParts(familyId)
    const caps = provider.capabilities?.[familyId]

    const details = [familyId, caps?.fast ? 'fast' : null, caps && !caps.reasoning ? 'no reasoning' : null]
      .filter(Boolean)
      .join(' · ')

    return (
      <label className="flex cursor-pointer items-center gap-2 px-3 py-1 text-xs hover:bg-(--ui-control-active-background)">
        <span
          className="block min-w-0 flex-1 overflow-hidden"
          onMouseOut={event => {
            if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
              return
            }

            setGo(false)
          }}
          onMouseOver={() => setGo(true)}
          title={details}
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <HoverScroll className="min-w-0 flex-1" go={go} marqueeKey={`name:${familyId}`}>
              <span data-row-label>
                <HighlightMatches query={terms} text={name} />
              </span>
            </HoverScroll>
          </span>
          <HoverScroll
            className="font-mono text-[0.65rem] font-normal text-(--ui-text-tertiary)"
            go={go}
            marqueeKey={`id:${familyId}`}
          >
            <HighlightMatches query={terms} text={familyId} />
          </HoverScroll>
        </span>
        <Switch checked={checked} onCheckedChange={onToggle} size="xs" />
      </label>
    )
  },
  (prev, next) =>
    prev.checked === next.checked &&
    prev.familyId === next.familyId &&
    prev.provider === next.provider &&
    prev.search === next.search &&
    prev.terms === next.terms
)

interface ModelVisibilityDialogProps {
  gw?: HermesGateway
  onOpenChange: (open: boolean) => void
  onOpenProviders: () => void
  open: boolean
  ownerConnectionId?: string
  profile?: string
  sessionId?: string | null
}

export function ModelVisibilityDialog({
  gw,
  onOpenChange,
  onOpenProviders,
  open,
  ownerConnectionId,
  profile = 'default',
  sessionId
}: ModelVisibilityDialogProps) {
  const { t } = useI18n()
  const copy = t.modelVisibility
  const [searchInput, setSearchInput] = useState('')
  // Filtering runs on the settled query; the input itself stays instant.
  const search = useDebouncedValue(searchInput, 150)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const stored = useStore($visibleModels)
  const collapsedProviders = useStore($collapsedProviders)

  // Order snapshot per open: toggling a switch must not reshuffle the list
  // under the pointer — enablement ranks freeze when the modal opens.
  const rankRef = useRef<Set<string> | null>(null)

  if (!open) {
    rankRef.current = null
  }

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId, ownerConnectionId),
    queryFn: (): Promise<ModelOptionsResponse> => requestModelOptions({ gateway: gw, profile, sessionId }),
    enabled: open
  })

  const providers = useMemo(
    () => (modelOptions.data?.providers ?? []).filter(provider => (provider.models ?? []).length > 0),
    [modelOptions.data]
  )

  const visible = effectiveVisibleKeys(stored, providers)

  if (open && rankRef.current === null) {
    rankRef.current = new Set(visible)
  }

  const rank = rankRef.current ?? visible

  const toggle = (provider: ModelOptionProvider, model: string) => {
    setVisibleModels(toggleModelVisibility($visibleModels.get(), providers, provider.slug, model))
  }

  const setProviderVisible = (provider: ModelOptionProvider, next: boolean) => {
    setVisibleModels(setProviderVisibility($visibleModels.get(), providers, provider.slug, next))
  }

  const q = normalize(search)

  // Token-AND index (`deepseek v4 flash opencode` finds
  // `opencode/deepseek-v4-flash` regardless of order), folded once per
  // catalog so keystrokes only scan cheap substrings. Highlighting reuses the
  // tokens only for multi-token queries — single-token queries keep the raw
  // string so separator folding highlights one contiguous range.
  const queryTokens = useMemo(() => searchFold(q).split(/\s+/).filter(Boolean), [q])
  const highlightQuery = queryTokens.length > 1 && /\s/.test(search) ? queryTokens : search

  const haystacks = useMemo(() => {
    const map = new Map<string, string>()

    for (const provider of providers) {
      for (const model of provider.models ?? []) {
        map.set(
          `${provider.slug}::${model}`,
          searchFold(`${model} ${provider.name} ${provider.slug} ${displayModelName(model)}`)
        )
      }
    }

    return map
  }, [providers])

  const matches = (provider: ModelOptionProvider, model: string) =>
    queryTokens.every(token => (haystacks.get(`${provider.slug}::${model}`) ?? '').includes(token))

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent bodyClassName="gap-0 overflow-hidden p-0" className="max-w-xs">
        <DialogHeader className="px-3 pb-1 pt-3">
          <DialogTitle className="text-[0.8125rem]">{copy.title}</DialogTitle>
        </DialogHeader>

        <div className="flex items-center gap-1.5 px-3 py-1.5">
          <Search className="pointer-events-none size-3.5 shrink-0 text-muted-foreground/70" />
          <input
            autoFocus
            className="h-5 w-full bg-transparent text-xs text-foreground placeholder:text-(--ui-text-tertiary) focus:outline-none"
            onChange={event => setSearchInput(event.target.value)}
            placeholder={copy.search}
            type="text"
            value={searchInput}
          />
        </div>

        <div className="max-h-[55vh] overflow-y-auto pb-1">
          {providers.length === 0 ? (
            <div className="px-3 py-5 text-center text-xs text-muted-foreground">
              {modelOptions.isPending ? <GlyphSpinner className="mx-auto text-sm" /> : copy.noAuthenticatedProviders}
            </div>
          ) : (
            providers.map(provider => {
              const allFamilies = collapseModelFamilies(provider.models ?? [])
              // A–Z, same order as the model menu; the raw id stays searchable
              // so an upstream prefix (`cmd/…`) filters even when the pretty
              // name hides it.
              // Enabled first, then A–Z — ranked by the open-time snapshot so
              // toggling never reshuffles.
              const enabledRank = (id: string) => (rank.has(modelVisibilityKey(provider.slug, id)) ? 0 : 1)

              const models = allFamilies
                .filter(family => matches(provider, family.id))
                .sort((a, b) => enabledRank(a.id) - enabledRank(b.id) || compareModelIds(a.id, b.id))

              if (models.length === 0) {
                return null
              }

              const shown = expanded.has(provider.slug) ? models : models.slice(0, DIALOG_RENDER_CAP)

              const onCount = allFamilies.filter(family =>
                visible.has(modelVisibilityKey(provider.slug, family.id))
              ).length

              const checkState = onCount === 0 ? false : onCount === allFamilies.length ? true : 'indeterminate'

              const collapsed = collapsedProviders.includes(provider.slug) && !q

              return (
                <div className="py-0.5" key={provider.slug}>
                  <div className="flex items-center gap-2 px-3 pb-0.5 pt-1">
                    <button
                      className="group/label flex w-full items-center gap-1 pb-0.5 pt-0.5 text-left text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary) hover:bg-transparent"
                      onClick={() => toggleCollapsedProvider(provider.slug)}
                      type="button"
                    >
                      <span className="min-w-0 truncate">
                        <HighlightMatches foldSeparators query={search} text={provider.name} />
                      </span>
                      <span className="shrink-0 font-normal normal-case tracking-normal">
                        {onCount}/{allFamilies.length}
                      </span>
                      <DisclosureCaret
                        className="shrink-0 opacity-0 transition group-hover/label:opacity-100"
                        open={!collapsed}
                        size="0.625rem"
                      />
                    </button>
                    <Checkbox
                      checked={checkState}
                      onCheckedChange={next => setProviderVisible(provider, next !== false)}
                    />
                  </div>
                  {!collapsed &&
                    shown.map(family => (
                      <VisibilityRow
                        checked={visible.has(modelVisibilityKey(provider.slug, family.id))}
                        familyId={family.id}
                        key={modelVisibilityKey(provider.slug, family.id)}
                        onToggle={() => toggle(provider, family.id)}
                        provider={provider}
                        search={search}
                        terms={highlightQuery}
                      />
                    ))}
                  {!collapsed && shown.length < models.length ? (
                    <button
                      className="w-full px-3 py-1 text-left text-[0.65rem] text-(--ui-text-tertiary) hover:text-foreground"
                      onClick={() =>
                        setExpanded(prev => {
                          const next = new Set(prev)
                          next.add(provider.slug)

                          return next
                        })
                      }
                      type="button"
                    >
                      Show all {models.length.toLocaleString()} — {models.length - shown.length} more
                    </button>
                  ) : null}
                </div>
              )
            })
          )}
        </div>

        <div className="px-3 py-2">
          <Button
            className="-ml-2 text-(--ui-text-tertiary)"
            onClick={() => {
              onOpenChange(false)
              onOpenProviders()
            }}
            size="xs"
            type="button"
            variant="text"
          >
            {copy.addProvider}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

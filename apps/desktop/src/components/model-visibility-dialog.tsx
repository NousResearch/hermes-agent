import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { Switch } from '@/components/ui/switch'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { Search } from '@/lib/icons'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { normalize } from '@/lib/text'
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

interface ModelVisibilityDialogProps {
  gw?: HermesGateway
  onOpenChange: (open: boolean) => void
  onOpenProviders: () => void
  open: boolean
  profile?: string
  sessionId?: string | null
}

export function ModelVisibilityDialog({
  gw,
  onOpenChange,
  onOpenProviders,
  open,
  profile = 'default',
  sessionId
}: ModelVisibilityDialogProps) {
  const { t } = useI18n()
  const copy = t.modelVisibility
  const [search, setSearch] = useState('')
  const stored = useStore($visibleModels)
  const collapsedProviders = useStore($collapsedProviders)

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId),
    queryFn: (): Promise<ModelOptionsResponse> => requestModelOptions({ gateway: gw, sessionId }),
    enabled: open
  })

  const providers = useMemo(
    () => (modelOptions.data?.providers ?? []).filter(provider => (provider.models ?? []).length > 0),
    [modelOptions.data]
  )

  const visible = effectiveVisibleKeys(stored, providers)

  const toggle = (provider: ModelOptionProvider, model: string) => {
    setVisibleModels(toggleModelVisibility($visibleModels.get(), providers, provider.slug, model))
  }

  const toggleProvider = (provider: ModelOptionProvider, next: boolean) => {
    setVisibleModels(setProviderVisibility($visibleModels.get(), providers, provider.slug, next))
  }

  const q = normalize(search)

  const matches = (provider: ModelOptionProvider, model: string) =>
    !q || `${model} ${provider.name} ${provider.slug} ${displayModelName(model)}`.toLowerCase().includes(q)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-xs gap-0 overflow-hidden p-0">
        <DialogHeader className="px-3 pb-1 pt-3">
          <DialogTitle className="text-[0.8125rem]">{copy.title}</DialogTitle>
        </DialogHeader>

        <div className="flex items-center gap-1.5 px-3 py-1.5">
          <Search className="pointer-events-none size-3.5 shrink-0 text-muted-foreground/70" />
          <input
            autoFocus
            className="h-5 w-full bg-transparent text-xs text-foreground placeholder:text-(--ui-text-tertiary) focus:outline-none"
            onChange={event => setSearch(event.target.value)}
            placeholder={copy.search}
            type="text"
            value={search}
          />
        </div>

        <div className="max-h-[55vh] overflow-y-auto pb-1">
          {providers.length === 0 ? (
            <div className="px-3 py-5 text-center text-xs text-muted-foreground">
              {modelOptions.isPending ? <GlyphSpinner className="mx-auto text-sm" /> : copy.noAuthenticatedProviders}
            </div>
          ) : (
            providers.map(provider => {
              const models = collapseModelFamilies(provider.models ?? []).filter(family => matches(provider, family.id))

              if (models.length === 0) {
                return null
              }

              const allFamilies = collapseModelFamilies(provider.models ?? [])

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
                        <HighlightMatches query={search} text={provider.name} />
                      </span>
                      <DisclosureCaret
                        className="shrink-0 opacity-0 transition group-hover/label:opacity-100"
                        open={!collapsed}
                        size="0.625rem"
                      />
                    </button>
                    <Checkbox checked={checkState} onCheckedChange={next => toggleProvider(provider, next !== false)} />
                  </div>
                  {!collapsed &&
                    models.map(family => {
                      const { name, tag } = modelDisplayParts(family.id)
                      const key = modelVisibilityKey(provider.slug, family.id)

                      return (
                        <label
                          className="flex cursor-pointer items-center gap-2 px-3 py-1 text-xs hover:bg-(--ui-control-active-background)"
                          key={key}
                        >
                          <span className="min-w-0 flex-1 truncate">
                            <HighlightMatches query={search} text={name} />
                            {tag ? <span className="text-(--ui-text-tertiary)"> {tag}</span> : null}
                          </span>
                          <Switch
                            checked={visible.has(key)}
                            onCheckedChange={() => toggle(provider, family.id)}
                            size="xs"
                          />
                        </label>
                      )
                    })}
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

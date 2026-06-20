import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Switch } from '@/components/ui/switch'
import type { HermesGateway } from '@/hermes'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import {
  $visibleModels,
  collapseModelFamilies,
  effectiveVisibleKeys,
  emptyProviderSentinelKey,
  isProviderSentinel,
  type ModelFamily,
  modelVisibilityKey,
  setVisibleModels
} from '@/store/model-visibility'
import type { ModelOptionProvider, ModelOptionsResponse } from '@/types/hermes'

interface ModelVisibilityDialogProps {
  gw?: HermesGateway
  onOpenChange: (open: boolean) => void
  onOpenProviders: () => void
  open: boolean
  sessionId?: string | null
}

export function ModelVisibilityDialog({
  gw,
  onOpenChange,
  onOpenProviders,
  open,
  sessionId
}: ModelVisibilityDialogProps) {
  const { t } = useI18n()
  const copy = t.modelVisibility
  const [search, setSearch] = useState('')
  const stored = useStore($visibleModels)

  const modelOptions = useQuery({
    queryKey: ['model-options', sessionId || 'global'],
    queryFn: (): Promise<ModelOptionsResponse> => {
      if (gw && sessionId) {
        return gw.request<ModelOptionsResponse>('model.options', { session_id: sessionId })
      }

      return getGlobalModelOptions()
    },
    enabled: open
  })

  const providers = useMemo(
    () => (modelOptions.data?.providers ?? []).filter(provider => (provider.models ?? []).length > 0),
    [modelOptions.data]
  )

  const visible = effectiveVisibleKeys(stored, providers)

  const toggle = (provider: ModelOptionProvider, model: string) => {
    const next = new Set(effectiveVisibleKeys($visibleModels.get(), providers))
    const key = modelVisibilityKey(provider.slug, model)
    const sentinel = emptyProviderSentinelKey(provider.slug)

    if (next.has(key)) {
      next.delete(key)

      // Check if this was the last real model for this provider.
      const remainingForProvider = [...next].some(k => k.startsWith(`${provider.slug}::`) && !isProviderSentinel(k))

      if (!remainingForProvider) {
        next.add(sentinel)
      }
    } else {
      next.delete(sentinel)
      next.add(key)
    }

    setVisibleModels(next)
  }

  const toggleProvider = (provider: ModelOptionProvider, models: ModelFamily[]) => {
    const next = new Set(effectiveVisibleKeys($visibleModels.get(), providers))
    const sentinel = emptyProviderSentinelKey(provider.slug)

    // Check if every model in the given (search-filtered) set is currently visible.
    const allVisible = models.every(family => next.has(modelVisibilityKey(provider.slug, family.id)))

    if (allVisible) {
      // Hide all in the set: remove each model key and store the sentinel.
      for (const family of models) {
        next.delete(modelVisibilityKey(provider.slug, family.id))
      }
      next.add(sentinel)
    } else {
      // Show all in the set: add each model key and clear the sentinel.
      next.delete(sentinel)
      for (const family of models) {
        next.add(modelVisibilityKey(provider.slug, family.id))
      }
    }

    setVisibleModels(next)
  }

  const q = search.trim().toLowerCase()

  const matches = (provider: ModelOptionProvider, model: string) =>
    !q || `${model} ${provider.name} ${provider.slug} ${displayModelName(model)}`.toLowerCase().includes(q)

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-xs gap-0 overflow-hidden p-0">
        <DialogHeader className="px-3 pb-1 pt-3">
          <DialogTitle className="text-[0.8125rem]">{copy.title}</DialogTitle>
        </DialogHeader>

        <div className="px-3 py-1.5">
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

              return (
                <div className="py-0.5" key={provider.slug}>
                  <div className="flex items-center gap-2 px-3 pb-0.5 pt-1">
                    <span className="min-w-0 flex-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
                      {provider.name}
                    </span>
                    <Switch
                      aria-label={`${provider.name} all models`}
                      checked={models.every(family =>
                        visible.has(modelVisibilityKey(provider.slug, family.id))
                      )}
                      onCheckedChange={() => toggleProvider(provider, models)}
                      size="xs"
                    />
                  </div>
                  {models.map(family => {
                    const { name, tag } = modelDisplayParts(family.id)
                    const key = modelVisibilityKey(provider.slug, family.id)

                    return (
                      <label
                        className="flex cursor-pointer items-center gap-2 px-3 py-1 text-xs hover:bg-accent/50"
                        key={key}
                      >
                        <span className="min-w-0 flex-1 truncate">
                          {name}
                          {tag ? <span className="text-(--ui-text-tertiary)"> {tag}</span> : null}
                        </span>
                        <Switch checked={visible.has(key)} onCheckedChange={() => toggle(provider, family.id)} />
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

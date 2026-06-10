import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { BrailleSpinner } from '@/components/ui/braille-spinner'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import type { HermesGateway } from '@/hermes'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import {
  $visibleModels,
  collapseModelFamilies,
  effectiveVisibleKeys,
  modelVisibilityKey,
  providerFamilyIds,
  toggleProviderVisibility,
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

    if (next.has(key)) {
      next.delete(key)
    } else {
      next.add(key)
    }

    setVisibleModels(next)
  }

  const toggleProvider = (provider: ModelOptionProvider) => {
    const currentStored = $visibleModels.get()
    let next = new Set(effectiveVisibleKeys(currentStored, providers))

    // Collect all family keys for this provider.
    const familyIds = providerFamilyIds(provider)
    const providerKeys = familyIds.map(id => modelVisibilityKey(provider.slug, id))

    const anyVisible = providerKeys.some(k => next.has(k))

    if (anyVisible) {
      // Disable ALL models of this provider; leave tombstone to prevent
      // effectiveVisibleKeys from re-adding defaults on reload.
      for (const k of providerKeys) {
        next.delete(k)
      }
      const tombstone = `${provider.slug}::`
      next.add(tombstone)
    } else {
      // Enable ALL models — remove tombstone if present, add all family keys.
      const tombstone = `${provider.slug}::`
      next.delete(tombstone)
      for (const k of providerKeys) {
        next.add(k)
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
              {modelOptions.isPending ? <BrailleSpinner className="mx-auto text-sm" /> : copy.noAuthenticatedProviders}
            </div>
          ) : (
            providers.map(provider => {
              const allFamilies = collapseModelFamilies(provider.models ?? [])

              if (allFamilies.length === 0) {
                return null
              }

              // Always show the provider header + toggle, even when all models
              // are disabled — otherwise user has no way to re-enable.
              const families = allFamilies.filter(family => matches(provider, family.id))

              return (
                <div className="py-0.5" key={provider.slug}>
                  {(() => {
                    const familyIds = providerFamilyIds(provider)
                    const anyVisible = familyIds.some(
                      id => visible.has(modelVisibilityKey(provider.slug, id))
                    )

                    return (
                      <label className="flex cursor-pointer items-center gap-2 px-3 pb-0.5 pt-1 hover:bg-accent/50">
                        <span className="text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
                          {provider.name}
                        </span>
                        <Switch
                          checked={anyVisible}
                          className="ml-auto"
                          onCheckedChange={() => toggleProvider(provider)}
                        />
                      </label>
                    )
                  })()}

                  {/* Only render model rows that pass search filter AND are visible */}
                  {families.map(family => {
                    const key = modelVisibilityKey(provider.slug, family.id)

                    if (!visible.has(key)) {
                      return null
                    }

                    const { name, tag } = modelDisplayParts(family.id)

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

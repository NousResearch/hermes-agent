import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { BrailleSpinner } from '@/components/ui/braille-spinner'
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
              const models = collapseModelFamilies(provider.models ?? []).filter(family => matches(provider, family.id))

              if (models.length === 0) {
                return null
              }

              return (
                <div className="py-0.5" key={provider.slug}>
                  <div className="px-3 pb-0.5 pt-1 text-[0.625rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
                    {provider.name}
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
          <button
            className="text-xs text-(--ui-text-tertiary) transition-colors hover:text-foreground"
            onClick={() => {
              onOpenChange(false)
              onOpenProviders()
            }}
            type="button"
          >
            {copy.addProvider}
          </button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

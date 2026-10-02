import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'

import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { RowButton } from '@/components/ui/row-button'
import { Switch } from '@/components/ui/switch'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { Plus, Search, X } from '@/lib/icons'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { foldIncludes, normalize } from '@/lib/text'
import { confirm } from '@/store/confirm'
import {
  $customModels,
  addCustomModel,
  customModelCandidate,
  isCustomModel,
  removeCustomModel,
  resetModelVisibilityKeepingCustoms,
  withCustomModels
} from '@/store/custom-models'
import {
  $visibleModels,
  collapseModelFamilies,
  effectiveVisibleKeys,
  modelVisibilityKey,
  seedKnownModels,
  setProviderVisibility,
  setVisibleModels,
  toggleModelVisibility
} from '@/store/model-visibility'

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
  const [search, setSearch] = useState('')
  const stored = useStore($visibleModels)
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null)
  const customModels = useStore($customModels)

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId, ownerConnectionId),
    queryFn: (): Promise<ModelOptionsResult> => requestModelOptions({ gateway: gw, profile, sessionId }),
    enabled: open
  })

  const providers = useMemo(
    () =>
      withCustomModels(
        (modelOptions.data?.providers ?? []).filter(provider => (provider.models ?? []).length > 0),
        customModels
      ),
    [modelOptions.data, customModels]
  )

  useEffect(() => seedKnownModels(providers), [providers])

  const visible = effectiveVisibleKeys(stored, providers)

  const toggle = (provider: ModelOptionProvider, model: string) => {
    setVisibleModels(toggleModelVisibility($visibleModels.get(), providers, provider.slug, model), providers)
  }

  const setProviderVisible = (provider: ModelOptionProvider, next: boolean) => {
    setVisibleModels(setProviderVisibility($visibleModels.get(), providers, provider.slug, next), providers)
  }

  const resetToDefaults = async () => {
    const ok = await confirm({
      confirmLabel: copy.resetAction,
      description: copy.resetDescription,
      destructive: true,
      title: copy.resetConfirm
    })

    if (ok) {
      resetModelVisibilityKeepingCustoms(providers)
    }
  }

  const q = normalize(search)

  const matches = (provider: ModelOptionProvider, model: string) =>
    !q || foldIncludes(`${model} ${provider.name} ${provider.slug} ${displayModelName(model)}`, q)

  // Typing an id no provider lists offers to add it — same gesture as the
  // pickers, minus the switch — and the new row lands visible. Only once the
  // query matches nothing, so a partial search isn't shadowed by the offer.
  const hasMatches = providers.some(provider =>
    collapseModelFamilies(provider.models ?? []).some(family => matches(provider, family.id))
  )

  const customSlug = hasMatches ? null : customModelCandidate(search, providers)
  const matchingProviders = providers.filter(provider =>
    collapseModelFamilies(provider.models ?? []).some(family => matches(provider, family.id))
  )
  const activeProvider = matchingProviders.find(provider => provider.slug === selectedProvider) ?? matchingProviders[0]

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent bodyClassName="gap-0 overflow-hidden p-0" className="max-w-2xl">
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

        <div className="flex min-h-0">
          {matchingProviders.length > 0 && (
            <div className="max-h-[55vh] w-2/5 shrink-0 overflow-y-auto p-2">
              {matchingProviders.map(provider => {
                const families = collapseModelFamilies(provider.models ?? [])
                const count = families.filter(family =>
                  visible.has(modelVisibilityKey(provider.slug, family.id))
                ).length

                return (
                  <RowButton
                    aria-pressed={activeProvider?.slug === provider.slug}
                    className="flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left text-xs hover:bg-(--ui-control-active-background) aria-pressed:bg-(--ui-control-active-background) focus-visible:outline-ring"
                    key={provider.slug}
                    onClick={() => setSelectedProvider(provider.slug)}
                  >
                    <span className="min-w-0 truncate">
                      <HighlightMatches foldSeparators query={search} text={provider.name} />
                    </span>
                    <span className="shrink-0 text-muted-foreground">
                      {count}/{families.length}
                    </span>
                  </RowButton>
                )
              })}
            </div>
          )}
          <div className="max-h-[55vh] min-w-0 flex-1 overflow-y-auto pb-1" key={activeProvider?.slug}>
            {providers.length === 0 ? (
              <div className="px-3 py-5 text-center text-xs text-muted-foreground">
                {modelOptions.isPending ? <GlyphSpinner className="mx-auto text-sm" /> : copy.noAuthenticatedProviders}
              </div>
            ) : (
              (activeProvider ? [activeProvider] : []).map(provider => {
                const models = collapseModelFamilies(provider.models ?? []).filter(family =>
                  matches(provider, family.id)
                )

                if (models.length === 0) {
                  return null
                }

                const allFamilies = collapseModelFamilies(provider.models ?? [])

                const onCount = allFamilies.filter(family =>
                  visible.has(modelVisibilityKey(provider.slug, family.id))
                ).length

                const checkState = onCount === 0 ? false : onCount === allFamilies.length ? true : 'indeterminate'

                return (
                  <div className="py-0.5" key={provider.slug}>
                    <div className="flex items-center gap-2 px-3 pb-0.5 pt-1">
                      <div className="group/label flex w-full items-center gap-1 pb-0.5 pt-0.5 text-left text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary) hover:bg-transparent">
                        <span className="min-w-0 truncate">
                          <HighlightMatches foldSeparators query={search} text={provider.name} />
                        </span>
                      </div>
                      <Checkbox
                        aria-label={provider.name}
                        checked={checkState}
                        onCheckedChange={next => setProviderVisible(provider, next !== false)}
                      />
                    </div>
                    {models.map(family => {
                      const { name, tag } = modelDisplayParts(family.id)
                      const key = modelVisibilityKey(provider.slug, family.id)

                      return (
                        <label
                          className="flex cursor-pointer items-center gap-2 px-3 py-1 text-xs hover:bg-(--ui-control-active-background)"
                          key={key}
                        >
                          <span className="min-w-0 flex-1 truncate">
                            <HighlightMatches foldSeparators query={search} text={name} />
                            {tag ? <span className="text-(--ui-text-tertiary)"> {tag}</span> : null}
                          </span>
                          {isCustomModel(customModels, provider.slug, family.id) && (
                            <Button
                              aria-label={copy.removeCustomModel}
                              className="-my-1 text-(--ui-text-tertiary)"
                              onClick={event => {
                                event.preventDefault()
                                removeCustomModel(provider.slug, family.id)
                              }}
                              size="icon-xs"
                              type="button"
                              variant="ghost"
                            >
                              <X className="size-3" />
                            </Button>
                          )}
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
            {customSlug && providers.length > 0 && (
              <div className="py-0.5">
                <div className="px-3 pb-0.5 pt-1 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">
                  {copy.addCustomModel}
                </div>
                {providers.map(provider => (
                  <RowButton
                    className="flex w-full items-center gap-2 px-3 py-1 text-left text-xs hover:bg-(--ui-control-active-background)"
                    key={`custom:${provider.slug}`}
                    onClick={() => {
                      addCustomModel(provider.slug, customSlug, provider)
                      setSearch('')
                    }}
                  >
                    <span className="min-w-0 flex-1 truncate">
                      {customSlug}
                      <span className="text-(--ui-text-tertiary)"> {provider.name}</span>
                    </span>
                    <Plus className="size-3 shrink-0 text-(--ui-text-tertiary)" />
                  </RowButton>
                ))}
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center justify-between px-3 py-2">
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
          {/* Only once there is something to undo, like the sidebar view reset. */}
          {stored !== null && modelOptions.isSuccess && (
            <Button className="-mr-2" onClick={() => void resetToDefaults()} size="xs" type="button" variant="text">
              {copy.resetToDefaults}
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

import { useStore } from '@nanostores/react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { HERMES_CONFIG_KEY } from '@/app/hooks/use-config-record'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { HighlightMatches } from '@/components/ui/highlight-matches'
import { Switch } from '@/components/ui/switch'
import { getHermesConfigRecord, type HermesGateway, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import {
  excludedProviderName,
  isProviderExcluded,
  readExcludedProviders,
  withExcludedProviders,
  withProviderExcluded
} from '@/lib/excluded-providers'
import { Search } from '@/lib/icons'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { displayModelName, modelDisplayParts } from '@/lib/model-status-label'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import {
  $visibleModels,
  collapseModelFamilies,
  effectiveVisibleKeys,
  modelVisibilityKey,
  setProviderVisibility,
  setVisibleModels,
  toggleModelVisibility
} from '@/store/model-visibility'
import { notifyError } from '@/store/notifications'
import { $collapsedProviders, toggleCollapsedProvider } from '@/store/provider-collapse'
import type { HermesConfigRecord, ModelOptionProvider, ModelOptionsResponse } from '@/types/hermes'

/** Config record backing the provider switches, keyed by profile like the model
 *  catalog above it: `GET /api/config` answers for whichever profile the app is
 *  routed to, and this dialog is mounted across profile switches — one shared
 *  key would paint the previous profile's blocklist. */
const configQueryKey = (profile: string) => ['hermes-config', 'excluded-providers', profile] as const

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
  const queryClient = useQueryClient()

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId),
    queryFn: (): Promise<ModelOptionsResponse> => requestModelOptions({ gateway: gw, sessionId }),
    enabled: open
  })

  const config = useQuery({
    queryKey: configQueryKey(profile),
    queryFn: getHermesConfigRecord,
    enabled: open
  })

  const providers = useMemo(
    () => (modelOptions.data?.providers ?? []).filter(provider => (provider.models ?? []).length > 0),
    [modelOptions.data]
  )

  const excluded = useMemo(() => readExcludedProviders(config.data), [config.data])

  // An excluded provider is absent from the catalog payload — the backend drops
  // it before the picker sees it — so its row is synthesized from the config
  // list. Without that, the switch would disappear along with the provider and
  // there'd be no way back short of hand-editing config.yaml. Only the slug is
  // known until it's re-enabled, which is why the row carries no models.
  const rows = useMemo(
    () => [
      // The catalog can still carry an excluded provider in one case: it's the
      // configured current provider, which the payload keeps so the UI can show
      // the saved selection. Read `off` from the config either way so the switch
      // never contradicts what's on disk.
      ...providers.map(provider => ({ off: isProviderExcluded(excluded, provider.slug), provider })),
      ...excluded
        .filter(slug => !providers.some(provider => provider.slug.toLowerCase() === slug.toLowerCase()))
        .map(slug => ({
          off: true,
          provider: { models: [], name: excludedProviderName(slug), slug } as ModelOptionProvider
        }))
    ],
    [providers, excluded]
  )

  const visible = effectiveVisibleKeys(stored, providers)

  const toggle = (provider: ModelOptionProvider, model: string) => {
    setVisibleModels(toggleModelVisibility($visibleModels.get(), providers, provider.slug, model))
  }

  const setProviderVisible = (provider: ModelOptionProvider, next: boolean) => {
    setVisibleModels(setProviderVisibility($visibleModels.get(), providers, provider.slug, next))
  }

  // Provider on/off is real config (`model_catalog.excluded_providers`), not a
  // local preference: the backend builds every picker's catalog from it, so an
  // off provider is gone from the composer menu, the ⌘-picker, the TUI and
  // `hermes model` alike — including one that authenticated itself from ambient
  // credentials (a logged-in `gh`, Claude Code's OAuth file).
  //
  // The switch flips optimistically (with rollback), but the record that gets
  // PERSISTED is re-read first: `PUT /api/config` takes a whole record, and the
  // cached one can be up to a staleTime old — edited meanwhile by a settings
  // page, the CLI, or another profile's backend. Writing the cached copy would
  // resurrect its stale values as an edit. The catalog is re-fetched afterwards
  // because the backend derives it from the config we just changed.
  const setProviderEnabled = async (provider: ModelOptionProvider, enabled: boolean) => {
    const cached = config.data
    const key = configQueryKey(profile)

    if (!cached) {
      return
    }

    queryClient.setQueryData<HermesConfigRecord>(
      key,
      withExcludedProviders(cached, withProviderExcluded(excluded, provider.slug, !enabled))
    )

    try {
      const fresh = await getHermesConfigRecord()

      const next = withExcludedProviders(
        fresh,
        withProviderExcluded(readExcludedProviders(fresh), provider.slug, !enabled)
      )

      await saveHermesConfig(next)
      queryClient.setQueryData<HermesConfigRecord>(key, next)
      // The settings pages cache the same record under their own key and save it
      // back whole — leave theirs stale and their next save would PUT a
      // `model_catalog` block from before this switch.
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: HERMES_CONFIG_KEY }),
        queryClient.invalidateQueries({ queryKey: ['model-options'] })
      ])
    } catch (err) {
      queryClient.setQueryData<HermesConfigRecord>(key, cached)
      notifyError(err, copy.providerToggleFailed)
    }
  }

  const q = normalize(search)

  const matches = (provider: ModelOptionProvider, model: string) =>
    !q || `${model} ${provider.name} ${provider.slug} ${displayModelName(model)}`.toLowerCase().includes(q)

  const providerMatches = (provider: ModelOptionProvider) =>
    !q || `${provider.name} ${provider.slug}`.toLowerCase().includes(q)

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
            onChange={event => setSearch(event.target.value)}
            placeholder={copy.search}
            type="text"
            value={search}
          />
        </div>

        <div className="max-h-[55vh] overflow-y-auto pb-1">
          {rows.length === 0 ? (
            <div className="px-3 py-5 text-center text-xs text-muted-foreground">
              {modelOptions.isPending ? <GlyphSpinner className="mx-auto text-sm" /> : copy.noAuthenticatedProviders}
            </div>
          ) : (
            rows.map(({ off, provider }) => {
              const allFamilies = collapseModelFamilies(provider.models ?? [])
              const models = allFamilies.filter(family => matches(provider, family.id))

              // An off provider has no models to match, so its row survives a
              // search on the provider itself; an on provider needs a model hit.
              if (off ? !providerMatches(provider) : models.length === 0) {
                return null
              }

              const onCount = allFamilies.filter(family =>
                visible.has(modelVisibilityKey(provider.slug, family.id))
              ).length

              const checkState = onCount === 0 ? false : onCount === allFamilies.length ? true : 'indeterminate'

              const collapsed = (collapsedProviders.includes(provider.slug) && !q) || off

              return (
                <div className="py-0.5" key={provider.slug}>
                  <div className="flex items-center gap-2 px-3 pb-0.5 pt-1">
                    <button
                      className={cn(
                        'group/label flex w-full items-center gap-1 pb-0.5 pt-0.5 text-left text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary) hover:bg-transparent',
                        off && 'opacity-50'
                      )}
                      disabled={off}
                      onClick={() => toggleCollapsedProvider(provider.slug)}
                      type="button"
                    >
                      <span className="min-w-0 truncate">
                        <HighlightMatches query={search} text={provider.name} />
                      </span>
                      {!off && (
                        <DisclosureCaret
                          className="shrink-0 opacity-0 transition group-hover/label:opacity-100"
                          open={!collapsed}
                          size="0.625rem"
                        />
                      )}
                    </button>
                    {!off && (
                      <Checkbox
                        checked={checkState}
                        onCheckedChange={next => setProviderVisible(provider, next !== false)}
                      />
                    )}
                    <Switch
                      aria-label={copy.providerToggle(provider.name)}
                      checked={!off}
                      disabled={!config.data}
                      onCheckedChange={next => void setProviderEnabled(provider, next)}
                      size="xs"
                    />
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

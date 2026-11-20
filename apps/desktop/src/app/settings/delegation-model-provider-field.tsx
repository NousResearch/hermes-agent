import type { ModelOptionProvider } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getGlobalModelOptions, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { findCatalogProvider } from '@/lib/model-options'
import { cn } from '@/lib/utils'
import { $customModels, withCustomModels } from '@/store/custom-models'

import { CONTROL_TEXT } from './constants'
import {
  CUSTOM_PROVIDER_VALUE,
  type DelegationModelProviderValue,
  type DelegationRouteDraft,
  DIRECT_VALUE,
  INHERIT_VALUE,
  MODEL_ONLY_VALUE,
  providerItem,
  useDelegationRouteDraft
} from './delegation-route-draft'
import { ModelSelect } from './model-select'
import { ListRow } from './primitives'

export type { DelegationModelProviderValue } from './delegation-route-draft'
export { INHERIT_VALUE, MODEL_ONLY_VALUE } from './delegation-route-draft'

interface DelegationModelProviderFieldProps {
  model: string
  provider: string
  baseUrl?: string
  modelDescription?: ReactNode
  scope?: ProfileScope
  onChange: (next: DelegationModelProviderValue) => void
}

const ROW_CONTROL = 'w-full min-w-0 max-w-full @2xl:w-72'

/** A route draft belongs to the same immutable owner as the config that seeded it. */
export function DelegationModelProviderField(props: DelegationModelProviderFieldProps) {
  const identity = JSON.stringify([profileScopeKey(props.scope), props.model, props.provider, props.baseUrl ?? ''])

  return <DelegationRouteEditor key={identity} {...props} />
}

/** Catalog rows for the owner being edited; custom ids enrich only providers returned for that owner. */
export function useOwnerCatalog(scope: ProfileScope | undefined) {
  const query = useQuery({
    queryKey: ['model-options', 'delegation', profileScopeKey(scope)],
    queryFn: () => getGlobalModelOptions(undefined, scope),
    retry: false
  })

  const customModels = useStore($customModels)

  const providers = useMemo(
    () =>
      withCustomModels(
        (query.data?.providers ?? []).filter(p => p.slug),
        customModels
      ),
    [query.data?.providers, customModels]
  )

  return { providers, query }
}

/** A saved provider the catalog lacks (alias, display name, custom) stays visible as-is. */
export function displayProviders(
  providers: readonly ModelOptionProvider[],
  draftProvider: string
): readonly ModelOptionProvider[] {
  if (!draftProvider || providers.some(p => p.slug === draftProvider)) {
    return providers
  }

  const found = findCatalogProvider(providers, draftProvider)

  return [{ name: found?.name ?? draftProvider, slug: draftProvider, models: found?.models ?? [] }, ...providers]
}

function DelegationRouteEditor({
  model,
  provider,
  baseUrl = '',
  modelDescription,
  scope,
  onChange
}: DelegationModelProviderFieldProps) {
  const { providers, query } = useOwnerCatalog(scope)
  const draft = useDelegationRouteDraft({ model, provider, baseUrl }, providers)
  const showsModel = draft.mode !== INHERIT_VALUE && !draft.useParentModel

  return (
    <div className="grid gap-1">
      <ProviderRow
        description={showsModel ? undefined : modelDescription}
        draft={draft}
        providers={displayProviders(providers, draft.provider)}
      />
      {showsModel && <ModelRow description={modelDescription} draft={draft} providers={providers} />}
      <div className="grid justify-items-end gap-3 pb-3 text-right">
        <RouteNotes draft={draft} />
        {query.isError && <CatalogRetry fetching={query.isFetching} onRetry={() => void query.refetch()} />}
        {draft.pending && <ApplyBar draft={draft} onApply={() => onChange(draft.value)} />}
      </div>
    </div>
  )
}

function ProviderRow({
  draft,
  providers,
  description
}: {
  draft: DelegationRouteDraft
  providers: readonly ModelOptionProvider[]
  description?: ReactNode
}) {
  const { t } = useI18n()
  const s = t.settings.model.subagents

  return (
    <ListRow
      action={
        <div className={cn('grid gap-2', ROW_CONTROL)}>
          <Select onValueChange={draft.chooseMode} value={draft.mode}>
            <SelectTrigger
              aria-label={s.providerLabel}
              className={cn(CONTROL_TEXT, '[&_[data-slot=select-value]]:truncate')}
            >
              <SelectValue placeholder={t.settings.model.provider} />
            </SelectTrigger>
            <SelectContent>
              {draft.hasDirectEndpoint && <SelectItem value={DIRECT_VALUE}>{s.direct}</SelectItem>}
              <SelectItem value={INHERIT_VALUE}>{s.inherit}</SelectItem>
              <SelectItem value={MODEL_ONLY_VALUE}>{s.modelOnly}</SelectItem>
              {providers.map(p => (
                <SelectItem key={p.slug} value={providerItem(p.slug)}>
                  {p.name}
                </SelectItem>
              ))}
              <SelectItem value={CUSTOM_PROVIDER_VALUE}>{s.customProvider}</SelectItem>
            </SelectContent>
          </Select>
          {draft.mode === CUSTOM_PROVIDER_VALUE && (
            <Input
              aria-label={s.customProviderLabel}
              onChange={event => draft.setProvider(event.target.value)}
              placeholder="custom:provider"
              value={draft.provider}
            />
          )}
        </div>
      }
      description={description}
      title={s.providerLabel}
    />
  )
}

function ModelRow({
  draft,
  providers,
  description
}: {
  draft: DelegationRouteDraft
  providers: readonly ModelOptionProvider[]
  description?: ReactNode
}) {
  const { t } = useI18n()
  const s = t.settings.model.subagents
  // Direct, model-only and hand-typed providers have no catalog list: free text only.
  const listed = draft.mode.startsWith(providerItem(''))
  const selected = listed ? findCatalogProvider(providers, draft.provider) : undefined

  return (
    <ListRow
      action={
        <ModelSelect
          aria-label={s.modelLabel}
          className={cn(ROW_CONTROL, '[&_[data-slot=select-value]]:truncate')}
          models={selected?.models ?? []}
          onValueChange={draft.setModel}
          provider={selected}
          providerSlug={listed ? draft.provider : ''}
          value={draft.model}
        />
      }
      description={description}
      title={s.modelLabel}
    />
  )
}

function RouteNotes({ draft }: { draft: DelegationRouteDraft }) {
  const { t } = useI18n()
  const s = t.settings.model.subagents
  const allowsParentModel = draft.mode !== INHERIT_VALUE && draft.mode !== MODEL_ONLY_VALUE
  const note = 'text-xs text-muted-foreground'

  return (
    <>
      {allowsParentModel && (
        <label className={cn('flex items-center gap-2', note)}>
          <Checkbox
            checked={draft.useParentModel}
            onCheckedChange={checked => draft.setUseParentModel(checked === true)}
          />
          {s.parentModel}
        </label>
      )}
      {draft.useParentModel && <p className={note}>{s.parentModelWarning}</p>}
      {draft.hasDirectEndpoint && (
        <p className={note}>{draft.mode === DIRECT_VALUE ? s.directActive : s.directClear}</p>
      )}
    </>
  )
}

function CatalogRetry({ fetching, onRetry }: { fetching: boolean; onRetry: () => void }) {
  const { t } = useI18n()

  return (
    <div className="flex flex-wrap items-center gap-2" role="status">
      <p className="text-xs text-muted-foreground">{t.settings.model.subagents.catalogFailed}</p>
      <Button disabled={fetching} onClick={onRetry} size="xs" variant="secondary">
        {t.common.retry}
      </Button>
    </div>
  )
}

function ApplyBar({ draft, onApply }: { draft: DelegationRouteDraft; onApply: () => void }) {
  const { t } = useI18n()

  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      <span className="text-xs text-muted-foreground">{t.settings.model.subagents.draftHint}</span>
      <Button onClick={draft.reset} size="sm" variant="text">
        {t.common.cancel}
      </Button>
      <Button disabled={!draft.canApply} onClick={onApply} size="sm">
        {t.common.apply}
      </Button>
    </div>
  )
}

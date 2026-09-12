import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import { getApiRequestConnection, type ProfileScope, profileScopeKey } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getGlobalModelOptions, getHermesConfigSchema, saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import type { HermesConfigRecord, ModelOptionProvider } from '@/types/hermes'

import { useHermesConfigRecord } from '../hooks/use-config-record'

import { DelegationModelProviderField } from './delegation-model-provider-field'
import { delegationModelsCopy, type DelegationModelsCopy } from './delegation-models-copy'
import {
  asRecord,
  delegationDraftChanged,
  delegationDraftValid,
  type DelegationFallbackMode,
  delegationModelsPatch,
  moveDelegationFallback,
  readDelegationModels,
  updateDelegationFallback
} from './delegation-models-state'

export function DelegationModelSettings({ scopeProfile }: { scopeProfile?: string }) {
  const gateway = useStore($gateway)
  const activeProfile = useStore($activeGatewayProfile)
  const { locale } = useI18n()
  const copy = delegationModelsCopy(locale)

  // Pin reads, catalog, write AND read-back to the same owner. Even the local
  // pool needs an explicit tag: omission can select a remote registry primary.
  const scope = useMemo(
    () => ({
      profile: scopeProfile ?? activeProfile ?? 'default',
      connectionId: getApiRequestConnection() ?? 'local'
    }),
    // gateway is the change signal for the ambient connection getter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [activeProfile, gateway, scopeProfile]
  )

  const config = useHermesConfigRecord(scope)

  const schema = useQuery({
    queryFn: () => getHermesConfigSchema(scope),
    queryKey: ['delegation-model-capabilities', profileScopeKey(scope)],
    retry: 1
  })

  const supported = schema.data?.capabilities?.delegation_fallbacks === true

  const catalog = useQuery({
    enabled: supported,
    queryFn: () => getGlobalModelOptions(undefined, scope),
    queryKey: ['delegation-model-options', profileScopeKey(scope)],
    retry: 1
  })

  if (!config.data || !schema.data) {
    const failed = config.isError || schema.isError

    return (
      <section aria-label={copy.title} className="my-6 grid gap-2">
        <h3 className="text-sm font-medium">{copy.title}</h3>
        <p className="text-xs text-muted-foreground">{failed ? copy.loadFailed : copy.loading}</p>
        {failed && (
          <Button
            onClick={() => {
              void config.refetch()
              void schema.refetch()
            }}
            size="sm"
          >
            {copy.refresh}
          </Button>
        )}
      </section>
    )
  }

  if (!supported) {
    return (
      <section aria-label={copy.title} className="my-6 grid gap-2">
        <h3 className="text-sm font-medium">{copy.title}</h3>
        <p className="text-xs text-muted-foreground">{schema.isError ? copy.loadFailed : copy.unsupported}</p>
        <Button onClick={() => void schema.refetch()} size="sm">
          {copy.refresh}
        </Button>
      </section>
    )
  }

  return (
    <DelegationModelEditor
      config={config.data}
      copy={copy}
      currentOwner={() =>
        (getApiRequestConnection() ?? 'local') === scope.connectionId &&
        (scopeProfile != null || ($activeGatewayProfile.get() ?? 'default') === scope.profile)
      }
      key={profileScopeKey(scope)}
      offline={catalog.isError}
      providers={catalog.data?.providers ?? []}
      reload={async () => {
        const next = await config.refetch()

        if (next.isError || !next.data) {
          throw new Error('Delegation config read-back failed')
        }

        return next.data
      }}
      retryCatalog={() => void catalog.refetch()}
      retrySchema={() => void schema.refetch()}
      runtimeReady={!schema.isError && !schema.isFetching}
      schemaFailed={schema.isError}
      scope={scope}
    />
  )
}

/** Local draft is not a saved setting: the full pair/chain commits only on Apply. */
function DelegationModelEditor({
  config,
  copy,
  currentOwner,
  offline,
  providers,
  reload,
  retryCatalog,
  retrySchema,
  runtimeReady,
  schemaFailed,
  scope
}: {
  config: HermesConfigRecord
  copy: DelegationModelsCopy
  currentOwner: () => boolean
  offline: boolean
  providers: readonly ModelOptionProvider[]
  reload: () => Promise<HermesConfigRecord>
  retryCatalog: () => void
  retrySchema: () => void
  runtimeReady: boolean
  schemaFailed: boolean
  scope: ProfileScope
}) {
  const [baseline, setBaseline] = useState(() => asRecord(config.delegation))
  const [draft, setDraft] = useState(() => readDelegationModels(config.delegation))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(false)
  const alive = useRef(true)
  const dirty = delegationDraftChanged(draft, baseline)
  const valid = delegationDraftValid(draft, baseline)
  const sourceSignature = JSON.stringify(asRecord(config.delegation))
  const baselineSignature = JSON.stringify(baseline)
  const conflict = sourceSignature !== baselineSignature && dirty && !saving
  const directEndpoint = !draft.clearEndpoint && !!baseline.base_url

  // eslint-disable-next-line no-restricted-syntax -- lifetime guard, not an atom mirror
  useEffect(() => {
    alive.current = true

    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    if (!dirty && !saving && sourceSignature !== baselineSignature) {
      const next = JSON.parse(sourceSignature) as Record<string, unknown>

      setBaseline(next)
      setDraft(readDelegationModels(next))
    }
  }, [baselineSignature, dirty, saving, sourceSignature])

  const reset = () => {
    setBaseline(asRecord(config.delegation))
    setDraft(readDelegationModels(config.delegation))
    setError(false)
  }

  const apply = async () => {
    if (!valid || !dirty || saving || conflict || !runtimeReady || !currentOwner()) {
      return
    }

    const patch = delegationModelsPatch(baseline, draft)

    setSaving(true)
    setError(false)

    try {
      const result = await saveHermesConfigRecord({ delegation: patch }, scope)

      if (!alive.current || !currentOwner()) {
        return
      }

      if (!result.ok) {
        throw new Error('Delegation config save failed')
      }

      const confirmed = await reload()

      if (!alive.current || !currentOwner()) {
        return
      }

      const stored = asRecord(confirmed.delegation)

      // An ACK alone does not prove the selected pair/chain survived persistence.
      const mismatch = Object.entries(patch).some(([key, value]) =>
        value == null ? stored[key] != null : JSON.stringify(stored[key]) !== JSON.stringify(value)
      )

      if (mismatch) {
        throw new Error('Delegation config read-back mismatch')
      }

      setBaseline(stored)
      setDraft(readDelegationModels(stored))
    } catch {
      if (alive.current && currentOwner()) {
        setError(true)
      }
    } finally {
      if (alive.current && currentOwner()) {
        setSaving(false)
      }
    }
  }

  return (
    <section aria-label={copy.title} className="my-6 grid gap-3">
      <div>
        <h3 className="text-sm font-medium">{copy.title}</h3>
        <p className="mt-1 text-xs text-muted-foreground">{copy.description}</p>
      </div>
      {schemaFailed && (
        <p className="text-xs text-muted-foreground" role="alert">
          {copy.loadFailed}
          <Button onClick={retrySchema} size="sm" variant="textStrong">
            {copy.refresh}
          </Button>
        </p>
      )}
      {offline && (
        <p className="text-xs text-muted-foreground" role="status">
          {copy.offline}
          <Button onClick={retryCatalog} size="sm" variant="textStrong">
            {copy.refresh}
          </Button>
        </p>
      )}
      {!offline && providers.length === 0 && <p className="text-xs text-muted-foreground">{copy.noProviders}</p>}
      <fieldset className="grid min-w-0 gap-3" disabled={saving}>
        {directEndpoint && <p className="text-xs text-muted-foreground">{copy.directHint}</p>}
        <DelegationModelProviderField
          copy={copy}
          directEndpoint={directEndpoint}
          model={draft.model}
          onChange={(pair, providerChanged) => {
            if (!saving) {
              setDraft(previous => ({ ...previous, ...pair, clearEndpoint: previous.clearEndpoint || providerChanged }))
            }
          }}
          parentProvider={String(asRecord(config.model).provider ?? '')}
          provider={draft.provider}
          providers={providers}
        />
        <div>
          <Button
            onClick={() => setDraft(previous => ({ ...previous, clearEndpoint: true, model: '', provider: '' }))}
            size="sm"
            variant="textStrong"
          >
            {copy.mainModel}
          </Button>
        </div>
        <Select
          onValueChange={mode => {
            if (!saving) {
              setDraft(previous => ({ ...previous, mode: mode as DelegationFallbackMode }))
            }
          }}
          value={draft.mode}
        >
          <SelectTrigger aria-label={copy.policy}>
            <SelectValue placeholder={copy.policy} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">{copy.auto}</SelectItem>
            <SelectItem value="inherit">{copy.inherit}</SelectItem>
            <SelectItem value="none">{copy.none}</SelectItem>
            <SelectItem value="custom">{copy.custom}</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {draft.mode === 'auto'
            ? copy.autoHint
            : draft.mode === 'inherit'
              ? copy.inheritHint
              : draft.mode === 'custom'
                ? copy.customHint
                : draft.mode === 'none'
                  ? copy.none
                  : copy.invalid}
        </p>
        {draft.mode === 'custom' && (
          <div className="grid gap-3">
            {draft.rows.map((row, index) => (
              <div className="grid gap-2" key={index}>
                <DelegationModelProviderField
                  allowInherit={false}
                  copy={copy}
                  labelPrefix={`${index + 1}. `}
                  model={row.model}
                  onChange={(pair, providerChanged) => {
                    if (!saving) {
                      setDraft(previous => ({
                        ...previous,
                        rows: previous.rows.map((entry, i) =>
                          i === index ? updateDelegationFallback(entry, pair, providerChanged) : entry
                        )
                      }))
                    }
                  }}
                  provider={row.provider}
                  providers={providers}
                />
                <div className="flex gap-1">
                  <Button
                    aria-label={`${copy.up} ${index + 1}`}
                    disabled={index === 0}
                    onClick={() =>
                      setDraft(previous => ({
                        ...previous,
                        rows: moveDelegationFallback(previous.rows, index, -1)
                      }))
                    }
                    size="sm"
                    variant="ghost"
                  >
                    ↑
                  </Button>
                  <Button
                    aria-label={`${copy.down} ${index + 1}`}
                    disabled={index === draft.rows.length - 1}
                    onClick={() =>
                      setDraft(previous => ({
                        ...previous,
                        rows: moveDelegationFallback(previous.rows, index, 1)
                      }))
                    }
                    size="sm"
                    variant="ghost"
                  >
                    ↓
                  </Button>
                  <Button
                    aria-label={`${copy.remove} ${index + 1}`}
                    onClick={() =>
                      setDraft(previous => ({
                        ...previous,
                        rows: previous.rows.filter((_, i) => i !== index)
                      }))
                    }
                    size="sm"
                    variant="ghost"
                  >
                    {copy.remove}
                  </Button>
                </div>
              </div>
            ))}
            <div>
              <Button
                onClick={() =>
                  setDraft(previous => ({
                    ...previous,
                    rows: [...previous.rows, { model: '', provider: '' }]
                  }))
                }
                size="sm"
                variant="textStrong"
              >
                {copy.add}
              </Button>
            </div>
          </div>
        )}
        {!valid && (
          <p className="text-xs text-muted-foreground" role="status">
            {copy.invalid}
          </p>
        )}
        {conflict && (
          <p className="text-xs text-muted-foreground" role="alert">
            {copy.conflict}
          </p>
        )}
        {error && (
          <p className="text-xs text-destructive" role="alert">
            {copy.failed}
          </p>
        )}
        <div className="flex gap-2">
          <Button disabled={!dirty || !valid || conflict || saving || !runtimeReady} onClick={() => void apply()} size="sm">
            {saving ? copy.saving : copy.apply}
          </Button>
          <Button disabled={!dirty && !conflict} onClick={reset} size="sm" variant="ghost">
            {conflict ? copy.refresh : copy.reset}
          </Button>
        </div>
      </fieldset>
    </section>
  )
}

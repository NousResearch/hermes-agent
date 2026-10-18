import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { FIELD_DESCRIPTIONS } from './constants'
import { DelegationModelProviderField, type DelegationModelProviderValue } from './delegation-model-provider-field'
import { fieldCopyForSchemaKey } from './field-copy'
import { getNested, setNested } from './helpers'

/** `delegation.provider` renders inside the `delegation.model` row, which applies both at once. */
export const DELEGATION_ROUTE_COMPANION_KEY = 'delegation.provider'

// A direct endpoint is URL + key + wire protocol; leaving it drops all three so a later endpoint
// never inherits a stale protocol (from #138048). Request settings belong to every route and stay.
const DIRECT_ENDPOINT_KEYS = ['delegation.base_url', 'delegation.api_key', 'delegation.api_mode']

/** The config after applying a subagent route. */
export function applyDelegationRoute(
  config: HermesConfigRecord,
  { model, provider, resetDirectEndpoint }: DelegationModelProviderValue
): HermesConfigRecord {
  let next = setNested(setNested(config, 'delegation.model', model), 'delegation.provider', provider)

  if (resetDirectEndpoint) {
    next = DIRECT_ENDPOINT_KEYS.reduce((acc, key) => setNested(acc, key, ''), next)
  }

  return next
}

/** One provider + model route draft for subagents, read from and written back to the owner's config. */
export function DelegationRouteRow({
  config,
  schema,
  scope,
  onConfigChange
}: {
  config: HermesConfigRecord
  schema: ConfigFieldSchema
  scope?: ProfileScope
  onConfigChange: (next: HermesConfigRecord) => void
}) {
  const { t } = useI18n()

  const description =
    fieldCopyForSchemaKey(t.settings.fieldDescriptions, 'delegation.model') ??
    fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, 'delegation.model') ??
    schema.description

  return (
    <div data-tour="field-delegation.model">
      <DelegationModelProviderField
        baseUrl={String(getNested(config, 'delegation.base_url') ?? '')}
        model={String(getNested(config, 'delegation.model') ?? '')}
        modelDescription={description}
        onChange={route => onConfigChange(applyDelegationRoute(config, route))}
        provider={String(getNested(config, 'delegation.provider') ?? '')}
        scope={scope}
      />
    </div>
  )
}

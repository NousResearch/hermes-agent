import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { capabilityScoped, type ProfileScope, profileScopeKey } from '@/api/client'
import { confirmProvider, readProviderOperation, reviewProvider } from '@/api/skill-catalog'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { $providerActions, needsProviderRecovery, type ProviderAction, providerActionKey } from '@/store/provider-actions'
import type { CatalogAction, CatalogOperation } from '@/types/skill-catalog'

import type { CatalogEntry } from './catalog-data'
import { PROVIDER_CATALOG_KEY } from './provider-catalog'

export function useProviderAction(profile: ProfileScope) {
  const pin = capabilityScoped(profile)
  const scope = profileScopeKey(pin)
  const [identifier, select] = useState<string | null>(null)
  const saved = useStore($providerActions)
  const state = identifier ? saved[providerActionKey(scope, identifier)] ?? null : null

  const save = (next: ProviderAction) => $providerActions.setKey(providerActionKey(scope, next.identifier), next)

  const refresh = () => {
    for (const key of ['skills-list', 'official-skills', 'skill-hub-sources', PROVIDER_CATALOG_KEY]) {
      void queryClient.invalidateQueries({ queryKey: [key] })
    }

    invalidateSlashCompletions()
  }

  // A reply belongs to the captured scope and operation, even after its view unmounts.
  const acceptOperation = (previous: ProviderAction, next: CatalogOperation) => {
    if (next.identifier === previous.identifier && next.action === previous.action && (!previous.operation || next.operation_id === previous.operation.operation_id)) { save({ ...previous, operation: next, busy: false, uncertain: false, error: undefined }) }
    else { save({ ...previous, busy: false, uncertain: true, error: 'invalid_response' });

 return }

    refresh()
  }

  const operation = useQuery({
    queryKey: ['skill-provider-operation', scope, state?.identifier, state?.operation?.operation_id],
    queryFn: () => readProviderOperation(pin, state!.identifier, state!.operation!.operation_id),
    enabled: Boolean(state?.operation && (state.operation.state === 'pending' || state.operation.state === 'unknown' || state.operation.settlement === 'reconciling')),
    refetchInterval: query => query.state.data && (query.state.data.state === 'completed' || query.state.data.state === 'failed') && query.state.data.settlement !== 'reconciling' ? false : 3000,
    retry: false
  })

  useEffect(() => {
    if (operation.data && state) { acceptOperation(state, operation.data) }
    // Only new server data settles an operation; render state is not authority.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [operation.data])

  const begin = async (entry: CatalogEntry, action: CatalogAction) => {
    if (state?.busy) { return }
    const previous = $providerActions.get()[providerActionKey(scope, entry.identifier)]
    select(entry.identifier)

    if (previous && needsProviderRecovery(previous)) { return }

    save({ identifier: entry.identifier, action, busy: true })

    try {
      const version = action === 'update' ? entry.management?.latest_version : entry.version
      const review = await reviewProvider(pin, entry.identifier, action, version || undefined)

      if (review.identifier !== entry.identifier || review.action !== action) { throw new Error('invalid_response') }

      save({ identifier: entry.identifier, action, busy: false, review })
    } catch (error) {
      save({ identifier: entry.identifier, action, busy: false, error: String(error) })
    }
  }

  const confirm = async (acknowledgements: string[]) => {
    const snapshot = identifier ? $providerActions.get()[providerActionKey(scope, identifier)] : undefined

    if (!snapshot?.review || snapshot.busy) { return }
    const exactAcknowledgements = snapshot.acknowledgements ?? acknowledgements
    const confirming = { ...snapshot, acknowledgements: exactAcknowledgements, busy: true, error: undefined }
    save(confirming)

    try {
      const result = await confirmProvider(pin, snapshot.review, exactAcknowledgements)

      if ('error' in result) { throw new Error(JSON.stringify(result.error)) }
      acceptOperation(confirming, result)
    } catch (error) {
      const message = String(error)
      const needsReview = /review_changed|review_expired|ownership_conflict|policy_denied|permission_denied|consent_required/.test(message)
      save({ ...confirming, busy: false, review: needsReview ? undefined : snapshot.review, uncertain: !needsReview, error: message })
    }
  }

  return { state, begin, confirm, operationError: operation.error, retryStatus: () => void operation.refetch(), close: () => { if (!state?.busy) { select(null) } } }
}

import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { type ProfileScope, profileScopeKey } from '@/api/client'
import { confirmProvider, readProviderOperation, reviewProvider } from '@/api/skill-catalog'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import type { CatalogAction, CatalogOperation, CatalogReview } from '@/types/skill-catalog'

import type { CatalogEntry } from './catalog-data'
import { PROVIDER_CATALOG_KEY } from './provider-catalog'

export interface ProviderAction {
  identifier: string
  action: CatalogAction
  review?: CatalogReview
  operation?: CatalogOperation
  busy: boolean
  uncertain?: boolean
  acknowledgements?: string[]
  error?: string
}

export function useProviderAction(profile: ProfileScope) {
  const [state, publish] = useState<ProviderAction | null>(null)
  const saved = useRef(new Map<string, ProviderAction>())

  const setState = (next: ProviderAction | null) => {
    if (next) { saved.current.set(next.identifier, next) }
    publish(next)
  }

  const current = useRef(state)
  current.current = state
  const alive = useRef(true)
  const inFlight = useRef(false)
  // eslint-disable-next-line no-restricted-syntax -- mount lifetime only; this ref never mirrors an atom
  useEffect(() => { alive.current = true;

 return () => { alive.current = false } }, [])

  const refresh = () => {
    for (const key of ['skills-list', 'official-skills', 'skill-hub-sources', PROVIDER_CATALOG_KEY]) {
      void queryClient.invalidateQueries({ queryKey: [key] })
    }

    invalidateSlashCompletions()
  }

  const acceptOperation = (next: CatalogOperation) => {
    if (!alive.current) { return }
    const previous = current.current

    if (previous && next.identifier === previous.identifier && next.action === previous.action && (!previous.operation || next.operation_id === previous.operation.operation_id)) { setState({ ...previous, operation: next, busy: false, uncertain: false, error: undefined }) }
    else if (previous) { setState({ ...previous, busy: false, uncertain: true, error: 'invalid_response' });

 return }

    refresh()
  }

  const operation = useQuery({
    queryKey: ['skill-provider-operation', profileScopeKey(profile), state?.identifier, state?.operation?.operation_id],
    queryFn: () => readProviderOperation(profile, state!.identifier, state!.operation!.operation_id),
    enabled: Boolean(state?.operation && (state.operation.state === 'pending' || state.operation.state === 'unknown' || state.operation.settlement === 'reconciling')),
    refetchInterval: query => query.state.data && (query.state.data.state === 'completed' || query.state.data.state === 'failed') && query.state.data.settlement !== 'reconciling' ? false : 3000,
    retry: false
  })

  useEffect(() => {
    if (operation.data) { acceptOperation(operation.data) }
    // Only new server data settles an operation; render state is not authority.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [operation.data])

  const begin = async (entry: CatalogEntry, action: CatalogAction) => {
    if (inFlight.current) { return }
    const previous = saved.current.get(entry.identifier)

    if (previous && (previous.uncertain || previous.operation?.state === 'pending' || previous.operation?.state === 'unknown' || previous.operation?.settlement === 'reconciling')) {
      setState(previous)

      return
    }

    inFlight.current = true
    setState({ identifier: entry.identifier, action, busy: true })

    try {
      const version = action === 'update' ? entry.management?.latest_version : entry.version
      const review = await reviewProvider(profile, entry.identifier, action, version || undefined)

      if (review.identifier !== entry.identifier || review.action !== action) { throw new Error('invalid_response') }

      if (alive.current) { setState({ identifier: entry.identifier, action, busy: false, review }) }
    } catch (error) {
      if (alive.current) { setState({ identifier: entry.identifier, action, busy: false, error: String(error) }) }
    } finally { inFlight.current = false }
  }

  const confirm = async (acknowledgements: string[]) => {
    const snapshot = current.current

    if (!snapshot?.review || inFlight.current) { return }
    inFlight.current = true
    setState({ ...snapshot, busy: true, error: undefined })

    try {
      const exactAcknowledgements = snapshot.acknowledgements ?? acknowledgements
      snapshot.acknowledgements = exactAcknowledgements
      const result = await confirmProvider(profile, snapshot.review, exactAcknowledgements)

      if ('error' in result) { throw new Error(JSON.stringify(result.error)) }
      acceptOperation(result)
    } catch (error) {
      if (!alive.current) { return }
      const message = String(error)
      const needsReview = /review_changed|review_expired|ownership_conflict|policy_denied|permission_denied|consent_required/.test(message)
      setState({ ...snapshot, busy: false, review: needsReview ? undefined : snapshot.review, uncertain: !needsReview, error: message })
    } finally { inFlight.current = false }
  }

  return { state, begin, confirm, operationError: operation.error, retryStatus: () => void operation.refetch(), close: () => { if (!inFlight.current) { setState(null) } } }
}

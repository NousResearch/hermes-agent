import { map } from 'nanostores'

import type { CatalogAction, CatalogOperation, CatalogReview } from '@/types/skill-catalog'

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

// Window-lifetime recovery authority, never discovery data or disk-persisted tokens.
// The identifier includes the provider's owner-qualified source id.
export const $providerActions = map<Record<string, ProviderAction | undefined>>({})

export const providerActionKey = (scope: string, identifier: string) => JSON.stringify([scope, identifier])

export const needsProviderRecovery = (state: ProviderAction) => Boolean(state.busy || state.uncertain || state.operation?.state === 'pending' || state.operation?.state === 'unknown' || state.operation?.settlement === 'reconciling')

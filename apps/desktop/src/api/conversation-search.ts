import { capabilityScoped, hermesApi, type ProfileScope, sessionReadOwnerPin } from './client'

export interface ConversationMatch {
  row_id: number
  role: string
  snippet: string
  timestamp: number
}

export interface ConversationSearchPage {
  session_id: string
  profile: string
  results: ConversationMatch[]
  pagination: { limit: number; offset: number; has_more: boolean; next_offset: number | null }
}

export const CONVERSATION_SEARCH_PAGE_SIZE = 50

export function searchConversation(storedId: string, scope: ProfileScope, query: string, offset = 0) {
  const route = {
    ...capabilityScoped(scope),
    ...(typeof scope === 'object' && scope?.connectionId === 'local' ? { connectionId: 'local' } : {}),
    ...sessionReadOwnerPin(storedId, scope)
  }

  const params = new URLSearchParams({ q: query, offset: String(offset), limit: String(CONVERSATION_SEARCH_PAGE_SIZE) })

  if (route.profile) {
    params.set('profile', route.profile)
  }

  return hermesApi<ConversationSearchPage>({
    ...route,
    path: `/api/sessions/${encodeURIComponent(storedId)}/messages/search?${params}`
  })
}

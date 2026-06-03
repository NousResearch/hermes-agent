import type { SessionInfo } from '@/types/hermes'

import { sessionTitle } from './chat-runtime'

export function sessionMatchesSearch(session: SessionInfo, query: string): boolean {
  const needle = query.trim().toLowerCase()

  if (!needle) {
    return true
  }

  return [
    session.id,
    session._lineage_root_id ?? '',
    sessionTitle(session),
    session.preview ?? '',
    session.cwd ?? ''
  ].some(value => value.toLowerCase().includes(needle))
}

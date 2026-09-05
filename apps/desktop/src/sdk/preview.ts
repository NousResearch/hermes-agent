import type { PluginSessionContext } from '@/contrib/session'
import { resolveSessionContributionContext } from '@/contrib/session-context'
import { openPreview } from '@/store/preview'
import { ownerLookupSessionRows, sessionMatchesStoredId } from '@/store/session'

import { safeViewerUrl } from '../../electron/plugin-viewer-policy'

export interface PluginPreviewInput {
  /** Client-reachable absolute HTTP(S) URL. No ambient auth headers are added. */
  url: string
  label?: string
  session: PluginSessionContext
}

export function currentPluginSession(session: PluginSessionContext): boolean {
  if (!session || !session.connectionId || !session.profile) {
    return false
  }

  const row = ownerLookupSessionRows().find(
    row =>
      sessionMatchesStoredId(row, session.storedSessionId || '') &&
      row.connection_id === session.connectionId &&
      row.profile === session.profile
  )

  const current = resolveSessionContributionContext({ storedSessionId: session.storedSessionId, row })

  return Boolean(
    current &&
    current.connectionId === session.connectionId &&
    current.profile === session.profile &&
    current.storedSessionId === session.storedSessionId &&
    current.runtimeSessionId === session.runtimeSessionId
  )
}

/** Explicit UI action, not an event handler. Ticket-bearing tabs never persist. */
export async function openPluginPreview(input: PluginPreviewInput): Promise<boolean> {
  const url = safeViewerUrl(input?.url)

  if (!url || !currentPluginSession(input.session)) {
    return false
  }

  openPreview(
    { kind: 'url', url, source: url, label: input.label || 'Viewer', transient: true, browserContext: 'isolated' },
    'explicit-link'
  )

  return true
}

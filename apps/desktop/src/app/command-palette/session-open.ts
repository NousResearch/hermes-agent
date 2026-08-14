import type { NavigateFunction } from 'react-router'

import { openSession, openSessionIntentFromModifiers } from '../open-session'
import { resolveSessionProfile } from '../session/hooks/use-session-actions/utils'

interface SessionOpenModifiers {
  ctrlKey?: boolean
  metaKey?: boolean
  shiftKey?: boolean
}

/** Open a palette session while preserving ownership for window intents. */
export async function openCommandPaletteSession(
  sessionId: string,
  profile: string | undefined,
  event: SessionOpenModifiers | undefined,
  navigate: NavigateFunction
): Promise<void> {
  const intent = openSessionIntentFromModifiers(event, 'stack')
  const owner = profile?.trim() || (intent === 'window' ? await resolveSessionProfile(sessionId) : undefined)

  openSession(sessionId, navigate, intent, owner)
}

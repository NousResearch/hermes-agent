import { activeItemIndexOwner, recordChatMessages } from '@/store/item-index'
import { $activeGatewayProfile } from '@/store/profile'
import { $portalShell } from '@/store/shell-variant'
import type { SessionMessage } from '@/types/hermes'

import type { GatewayEventContext } from './types'

/** The message a finished reply or tool call adds to the transcript, in the shape the Library's
 *  extractor reads; null for every other event. */
function streamedMessage({ event, payload, occurredAt }: GatewayEventContext): null | SessionMessage {
  if (event.type === 'message.complete') {
    const text = typeof payload?.text === 'string' ? payload.text : ''

    return text ? { role: 'assistant', content: text, timestamp: occurredAt } : null
  }

  if (event.type === 'tool.complete' && typeof payload?.name === 'string') {
    const result = payload.result
    const content = typeof result === 'string' ? result : JSON.stringify(result ?? null)

    return { role: 'tool', tool_name: payload.name, content, timestamp: occurredAt }
  }

  return null
}

/** Records what a chat streams into the item index, so the Library and the items panel show it
 *  without reading the transcript again (store/item-index.ts). The redesign reads the index;
 *  classic does not, so it records nothing. */
export function recordStreamedItems(ctx: GatewayEventContext): void {
  const { deps, event, sessionId } = ctx
  const message = sessionId && $portalShell.get() ? streamedMessage(ctx) : null
  const storedSessionId = sessionId ? deps.sessionStateByRuntimeIdRef.current.get(sessionId)?.storedSessionId : null

  if (!message || !storedSessionId) {
    return
  }

  recordChatMessages(
    {
      connectionId: event.connectionId ?? activeItemIndexOwner(),
      profile: event.profile || $activeGatewayProfile.get() || 'default',
      sessionId: storedSessionId
    },
    [message]
  )
}

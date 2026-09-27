import type { SessionHistoryResult } from '@hermes/shared'

import { segmentTranscriptDirectives } from '@/lib/transcript-directives'
import { requestGatewayForAgent } from '@/store/gateway'

/**
 * The handoff directive as the backend persisted it for the welcome chat. The build session is named and seeded
 * from these attrs, so they come from the stored reply, not the renderer's streamed copy of it: on Windows the
 * streamed copy once repeated its own chunks ("Set up mySet up my games…") while the stored reply was intact, and
 * the handoff carried the garbled task and brief into the new session. Null when no persisted reply holds one.
 */
export async function readPersistedHandoff(
  connectionId: null | string,
  profile: string,
  runtimeId: string
): Promise<null | Readonly<Record<string, string>>> {
  const history = await requestGatewayForAgent<Partial<SessionHistoryResult>>(
    connectionId,
    profile,
    'session.history',
    {
      session_id: runtimeId
    }
  )

  const messages = Array.isArray(history?.messages) ? history.messages : []

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]

    if (message?.role !== 'assistant' || message.text == null) {
      continue
    }

    for (const segment of segmentTranscriptDirectives(message.text) ?? []) {
      if (
        segment.kind === 'directive' &&
        segment.directive.name === 'onboarding' &&
        segment.directive.attrs.step === 'handoff'
      ) {
        return segment.directive.attrs
      }
    }
  }

  return null
}

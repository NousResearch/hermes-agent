import type { SessionHistoryResult } from '@hermes/shared'

import { segmentTranscriptDirectives } from '@/lib/transcript-directives'
import { requestGatewayForAgent } from '@/store/gateway'

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

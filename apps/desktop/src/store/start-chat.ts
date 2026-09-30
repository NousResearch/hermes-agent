import { parseMaybeObject } from '@/components/assistant-ui/tool/fallback-model/format'
import { $focusedStoredSessionId, isSessionInForeground } from '@/store/session-states'

export type StartChatOutcome =
  | { profile: string; sessionId: string; status: 'started'; title: null | string }
  | { reason: string; status: 'rejected' }

export function readStartChatResult(result: unknown): null | StartChatOutcome {
  const row = parseMaybeObject(result)

  if (row.status === 'started' && typeof row.session_id === 'string' && row.session_id) {
    return {
      profile: typeof row.profile === 'string' ? row.profile : '',
      sessionId: row.session_id,
      status: 'started',
      title: typeof row.title === 'string' && row.title.trim() ? row.title.trim() : null
    }
  }

  if (row.status === 'rejected') {
    return { reason: typeof row.reason === 'string' ? row.reason : '', status: 'rejected' }
  }

  return null
}

const liveStarts = new Set<string>()

export function markLiveStartChat(toolCallId: string): void {
  liveStarts.add(toolCallId)
}

export function takeLiveStartChat(toolCallId: string): boolean {
  return liveStarts.delete(toolCallId)
}

export function isStartChatCallerWatched(storedId: string): boolean {
  return $focusedStoredSessionId.get() !== null && isSessionInForeground(storedId)
}

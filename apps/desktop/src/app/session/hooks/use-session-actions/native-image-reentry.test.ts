import { afterEach, expect, it, vi } from 'vitest'

import { reconcileActiveTranscript } from '@/app/contrib/hooks/use-background-sync'
import { getLatestSessionMessages } from '@/hermes'
import { type ChatMessage, textPart, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import type { SessionMessage, SessionMessagesResponse } from '@/types/hermes'

vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn()
}))

// SYNTHETIC TEST FIXTURES ONLY. No live history, image bytes or owner RPC.
const SID = 'synthetic-reentry-runtime'
const STORED = 'synthetic-reentry-stored'
const caption = 'Synthetic fixture: inspect this panel.'
const nativeHint = `${caption}\n\n[Image attached at: /synthetic/reentry.png]`

const prefix: SessionMessage[] = [
  { id: 101, role: 'user', timestamp: 101, content: 'Synthetic earlier prompt.' },
  { id: 102, role: 'assistant', timestamp: 102, content: 'Synthetic earlier reply.' }
]

const localPrompt: ChatMessage = {
  id: 'user-synthetic-retained',
  role: 'user',
  timestamp: 103,
  parts: [textPart(caption)],
  attachmentRefs: ['data:image/png;base64,SYNTHETIC-PREVIEW-NOT-AN-IMAGE']
}

const localReply: ChatMessage = {
  id: 'assistant-stream-synthetic-retained',
  role: 'assistant',
  pending: false,
  parts: [textPart('Synthetic last reply.')]
}

afterEach(() => {
  clearAllSessionStates()
  vi.resetAllMocks()
})

function storedPage(representation: 'flattened' | 'multipart'): SessionMessagesResponse {
  return {
    session_id: STORED,
    messages: [
      ...prefix,
      {
        id: 153,
        role: 'user',
        timestamp: 103,
        content:
          representation === 'flattened'
            ? `${nativeHint}\n[screenshot]`
            : [
                { type: 'text', text: nativeHint },
                { type: 'image_url', image_url: { url: `data:image/png;base64,${'A'.repeat(64)}` } }
              ]
      },
      { id: 154, role: 'assistant', timestamp: 104, content: 'Synthetic last reply.' }
    ]
  }
}

async function heldReentry(page: SessionMessagesResponse, retained: ChatMessage[]) {
  let release!: (page: SessionMessagesResponse) => void

  const response = new Promise<SessionMessagesResponse>(resolve => {
    release = resolve
  })

  vi.mocked(getLatestSessionMessages).mockReturnValueOnce(response)
  let state = { ...createClientSessionState(STORED), messages: retained }
  publishSessionState(SID, state)

  const refresh = reconcileActiveTranscript({
    // The prior runtime has just been selected again. The retained view is
    // already painted; return/focus refresh now waits on its stored page.
    activeSessionIdRef: { current: SID },
    selectedStoredSessionIdRef: { current: STORED },
    busyRef: { current: false },
    requestSequenceRef: { current: 0 },
    signatureRef: { current: new Map() },
    resolveSession: () => ({ profile: 'default' }),
    updateSessionState: (_, update) => {
      state = update(state)
      publishSessionState(SID, state)

      return state
    }
  })

  try {
    expect(getLatestSessionMessages).toHaveBeenCalledOnce()
    expect(state.messages).toBe(retained)
  } finally {
    // Always release, including assertion failure, so teardown cannot hang.
    release(page)
    await refresh
  }

  return state.messages
}

it.each(['flattened', 'multipart'] as const)(
  'does not append a retained older image prompt when a held re-entry history read returns (%s)',
  async representation => {
    const retained = [...toChatMessages(prefix), localPrompt, localReply]
    const messages = await heldReentry(storedPage(representation), retained)
    expect(messages.filter(message => message.role === 'user')).toHaveLength(2)
    expect(messages.at(-1)?.rowId).toBe(154)
    expect(messages.at(-1)?.role).toBe('assistant')
  }
)

it('preserves a genuinely newer repeat with the same caption and its own durable row identity', async () => {
  const earlier = toChatMessages(storedPage('flattened').messages)
  const repeat: ChatMessage = { ...localPrompt, id: 'user-synthetic-new-repeat', rowId: 200 }
  const messages = await heldReentry(storedPage('multipart'), [...earlier, repeat])
  expect(messages.filter(message => message.role === 'user')).toHaveLength(3)
  expect(messages.at(-1)?.rowId).toBe(200)
})

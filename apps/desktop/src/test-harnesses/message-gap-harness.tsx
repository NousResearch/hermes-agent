import '@/styles.css'

import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { ChatBar } from '@/app/chat/composer'
import type { ChatBarState } from '@/app/chat/composer/types'
import { isRouteSessionMismatch } from '@/app/chat/route-session-state'
import { routedSessionIsLoading } from '@/app/chat/thread-loading'
import { runningProjectionStreamId } from '@/app/session/hooks/use-session-actions/utils'
import { Thread } from '@/components/assistant-ui/thread'
import { TranscriptWindowProvider } from '@/components/assistant-ui/thread/transcript-window'
import type { ChatMessage } from '@/lib/chat-messages'

const at = new Date('2026-08-30T21:29:20.000-07:00')
const RUNTIME_ID = 'runtime-gap'
const STORED_ID_BEFORE = 'stored-gap-before-compaction'
const STORED_ID_AFTER = 'stored-gap-after-compaction'
const GAP_ID = 'assistant-stream-runtime-gap'
const CONTINUITY_MODE = new URLSearchParams(window.location.search).get('mode') === 'continuity'

type UserThreadMessage = Extract<ThreadMessage, { role: 'user' }>
type AssistantThreadMessage = Extract<ThreadMessage, { role: 'assistant' }>

const user = (id: string, text: string): UserThreadMessage => ({
  attachments: [],
  id,
  role: 'user',
  content: [{ type: 'text', text }],
  createdAt: at,
  metadata: { custom: {} }
})

const assistant = (
  id: string,
  content: AssistantThreadMessage['content'],
  status: AssistantThreadMessage['status'] = { type: 'complete', reason: 'stop' }
): AssistantThreadMessage => ({
  id,
  role: 'assistant',
  content,
  createdAt: at,
  metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null },
  status
})

const historyTurn = (index: number): ThreadMessage[] => [
  user(`user-history-${index}`, `History user ${index}`),
  assistant(`assistant-history-${index}`, [
    {
      type: 'text',
      text: CONTINUITY_MODE
        ? `History assistant ${index}. ${'detail '.repeat(1_150)}`
        : Array.from({ length: 18 }, (_, line) => `History assistant ${index} line ${line + 1}.`).join('\n\n')
    }
  ])
]

const initialMessages = (): ThreadMessage[] => {
  const resumeProjection: ThreadMessage[] = [
    ...Array.from({ length: CONTINUITY_MODE ? 38 : 4 }, (_, index) => historyTurn(index + 1)).flat(),
    user('user-live', 'Investigate the message gap and scroll jump.'),
    assistant('assistant-scaffold-before', [
      {
        type: 'reasoning',
        text: 'Explored five relevant files and compared the transcript producers.'
      }
    ]),
    // Production resume projects an empty pending assistant boundary for the
    // correction that is still running. The resumed state must adopt this id
    // before the first reasoning delta arrives.
    assistant(GAP_ID, [], { type: 'running' })
  ]

  const streamId = runningProjectionStreamId(
    resumeProjection.map(
      message =>
        ({
          id: message.id,
          pending: message.status?.type === 'running',
          parts: message.content,
          role: message.role,
          timestamp: message.createdAt?.getTime()
        }) as ChatMessage
    ),
    true
  )

  const reasoning = [{ type: 'reasoning' as const, text: 'Tracing the live stream row after resume.' }]

  if (!streamId) {
    // The pre-fix ownership failure: a new stream id leaves the projected
    // assistant shell mounted between completed and active scaffold rows.
    return [...resumeProjection, assistant('assistant-stream-event-gap', reasoning, { type: 'running' })]
  }

  return resumeProjection.map(message =>
    message.id === streamId ? ({ ...message, content: reasoning } as ThreadMessage) : message
  )
}

const composerState: ChatBarState = {
  model: { canSwitch: false, model: 'test-model', provider: 'test-provider' },
  tools: { enabled: false, label: 'Tools' },
  voice: { active: false, enabled: false }
}

declare global {
  interface Window {
    messageGapHarness: {
      collapseGap: () => void
      completeBackfill: () => void
      messages: () => Array<{ contentLength: number; id: string; role: string; status: string }>
      pagination: () => { hasMoreBefore: boolean; nextOffset: number }
      routeState: () => {
        loadingSession: boolean
        mismatch: boolean
        paneId: string
        routedId: string
        selectedId: string
        showChatBar: boolean
      }
      reset: () => void
    }
  }
}

function Harness() {
  const [messages, setMessages] = useState<ThreadMessage[]>(initialMessages)
  const [selectedStoredId, setSelectedStoredId] = useState(STORED_ID_BEFORE)
  const [hasMoreBefore, setHasMoreBefore] = useState(true)

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: true,
    messages,
    onCancel: async () => undefined,
    onNew: async () => undefined
  })

  const expandWindow = useCallback(() => undefined, [])

  const transcriptWindow = useMemo(
    () => ({ expandWindow, olderAvailable: hasMoreBefore }),
    [expandWindow, hasMoreBefore]
  )

  const routeMismatch = isRouteSessionMismatch(STORED_ID_BEFORE, selectedStoredId, [
    { _lineage_root_id: STORED_ID_BEFORE, id: selectedStoredId }
  ])

  const loadingSession = routedSessionIsLoading({
    activeSessionId: RUNTIME_ID,
    knownHistory: true,
    messagesEmpty: messages.length === 0,
    resumeExhausted: false,
    routeSessionMismatch: routeMismatch,
    routedSessionView: true
  })

  const showChatBar = !loadingSession

  useEffect(() => {
    window.messageGapHarness = {
      collapseGap: () => {
        // One React commit models the observed compaction/resume boundary: the
        // stale empty projection is gone while the stored tip rotates under the
        // same live runtime. No imperative DOM removal is involved.
        setMessages(current => current.filter(message => message.id !== GAP_ID || message.content.length > 0))
        setSelectedStoredId(STORED_ID_AFTER)
      },
      completeBackfill: () => setHasMoreBefore(false),
      messages: () =>
        messages.map(message => ({
          contentLength: message.content.length,
          id: message.id,
          role: message.role,
          status: message.status?.type ?? 'unknown'
        })),
      pagination: () => ({ hasMoreBefore, nextOffset: 120 }),
      routeState: () => ({
        loadingSession,
        mismatch: routeMismatch,
        paneId: 'workspace',
        routedId: STORED_ID_BEFORE,
        selectedId: selectedStoredId,
        showChatBar
      }),
      reset: () => window.location.reload()
    }
  }, [hasMoreBefore, loadingSession, messages, routeMismatch, selectedStoredId, showChatBar])

  return (
    <MemoryRouter>
      <AssistantRuntimeProvider runtime={runtime}>
        <TranscriptWindowProvider value={transcriptWindow}>
          <main
            className="relative h-[520px] w-[720px] overflow-hidden bg-background text-foreground"
            data-pane-id="workspace"
          >
            <div className="h-[390px] overflow-hidden">
              <Thread sessionId={RUNTIME_ID} sessionKey="lineage-root-gap" />
            </div>
            {showChatBar && (
              <ChatBar
                busy
                disabled={false}
                focusKey={RUNTIME_ID}
                onCancel={() => undefined}
                onSubmit={() => false}
                queueSessionKey="lineage-root-gap"
                sessionId={RUNTIME_ID}
                state={composerState}
              />
            )}
          </main>
        </TranscriptWindowProvider>
      </AssistantRuntimeProvider>
    </MemoryRouter>
  )
}

createRoot(document.getElementById('root')!).render(<Harness />)

import { useEffect, useMemo, useReducer, useRef } from 'react'

import { isSubmitEnter } from '@/lib/ime'
import {
  initialSideChatState,
  type SideChatEvent,
  sideChatReducer,
  type SideChatState,
  type SideChatTurn
} from '@/store/side-chat'

/**
 * The side chat — the whole renderer surface of the floating `/btw` window.
 *
 * All behavior rides `sideChatReducer` (pure, unit-tested): a blank submit
 * sends nothing and clears nothing, a submit before the parent conversation is
 * known sends nothing, concurrent asides are allowed because each is its own
 * backend side agent, and replies are matched by ask id so two in flight can
 * finish in any order.
 *
 * The window has NO gateway connection. The conversation it asks about arrives
 * over IPC (`onContext`) and its questions go back the same road to the
 * renderer that ran `/btw`, which calls `prompt.btw` — see store/side-chat.
 *
 * Answers are rendered as PLAIN TEXT, deliberately. A `/btw` answer is a short
 * aside about the chat next to it, and pulling the full markdown/assistant-ui
 * stack into a window that has no app shell would buy formatting at the cost of
 * everything that stack expects to find around it.
 */
export function SideChatApp() {
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const threadRef = useRef<HTMLDivElement>(null)

  // The reducer returns { ask, state }; this wrapper performs the side effect
  // (hand the ask to the shell) and stores the next state, so the decision
  // stays pure and testable while the effects stay in one place.
  const [state, dispatch] = useReducer((current: SideChatState, event: SideChatEvent) => {
    const { ask, state: next } = sideChatReducer(current, event)

    if (ask) {
      window.hermesDesktop?.sideChat?.ask(ask)
    }

    return next
  }, initialSideChatState)

  useEffect(() => {
    const api = window.hermesDesktop?.sideChat

    const offContext = api?.onContext(context => {
      if (context?.sessionId) {
        dispatch({
          context: {
            question: typeof context.question === 'string' ? context.question : '',
            sessionId: context.sessionId,
            title: typeof context.title === 'string' ? context.title : ''
          },
          type: 'context'
        })
        requestAnimationFrame(() => composerRef.current?.focus())
      }
    })

    const offReply = api?.onReply(reply => {
      if (reply?.askId) {
        dispatch({
          reply: {
            askId: reply.askId,
            error: typeof reply.error === 'string' ? reply.error : '',
            text: typeof reply.text === 'string' ? reply.text : ''
          },
          type: 'reply'
        })
      }
    })

    composerRef.current?.focus()

    return () => {
      offContext?.()
      offReply?.()
    }
  }, [])

  // Follow the newest exchange. Asides are short, so pinning to the bottom is
  // always what the reader wants — there is no long backlog to lose your place in.
  useEffect(() => {
    const thread = threadRef.current

    if (thread) {
      thread.scrollTop = thread.scrollHeight
    }
  }, [state.turns])

  const subtitle = useMemo(() => {
    if (!state.context) {
      return 'Connecting…'
    }

    return state.context.title || 'this conversation'
  }, [state.context])

  const send = () => dispatch({ askId: newAskId(), type: 'submit' })
  const ready = Boolean(state.context)

  return (
    <div style={SHEET}>
      <header style={HEADER}>
        <div style={{ minWidth: 0 }}>
          <div style={TITLE}>Side chat</div>
          <div style={SUBTITLE} title={subtitle}>
            about {subtitle}
          </div>
        </div>
        <button
          aria-label="Close side chat"
          onClick={() => window.hermesDesktop?.sideChat?.close()}
          style={CLOSE_BUTTON}
          type="button"
        >
          ✕
        </button>
      </header>

      <div ref={threadRef} style={THREAD}>
        {state.turns.length === 0 ? (
          <p style={EMPTY}>
            Ask anything about this conversation. Answers come from a snapshot, so the chat itself keeps going
            untouched.
          </p>
        ) : (
          state.turns.map(turn => <Turn key={turn.askId} turn={turn} />)
        )}
      </div>

      <div style={COMPOSER_ROW}>
        <textarea
          aria-label="Ask a side question"
          disabled={!ready}
          onChange={event => dispatch({ draft: event.target.value, type: 'edit' })}
          onKeyDown={event => {
            if (isSubmitEnter(event) && !event.shiftKey) {
              event.preventDefault()
              send()
            }
          }}
          placeholder={ready ? 'Ask a side question…' : 'Waiting for the conversation…'}
          ref={composerRef}
          rows={2}
          spellCheck={false}
          style={COMPOSER}
          value={state.draft}
        />
      </div>
    </div>
  )
}

function Turn({ turn }: { turn: SideChatTurn }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={QUESTION}>{turn.question}</div>
      {turn.pending ? (
        <div style={PENDING}>Thinking…</div>
      ) : turn.error ? (
        <div style={ERROR}>{turn.error}</div>
      ) : (
        <div style={ANSWER}>{turn.answer}</div>
      )}
    </div>
  )
}

// Unique per window, which is all the reducer needs: ask ids never leave this
// renderer except to come straight back on the matching reply.
let askCounter = 0
const newAskId = () => `ask-${Date.now().toString(36)}-${(askCounter += 1)}`

const SHEET: React.CSSProperties = {
  background: 'var(--ui-bg-elevated, var(--background))',
  border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.35))',
  borderRadius: 12,
  boxShadow: '0 18px 48px rgba(0,0,0,0.38)',
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  overflow: 'hidden',
  width: '100vw'
}

const HEADER: React.CSSProperties = {
  alignItems: 'center',
  borderBottom: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.25))',
  display: 'flex',
  flexShrink: 0,
  gap: 8,
  justifyContent: 'space-between',
  padding: '8px 10px 8px 14px',
  // Frameless window: the header IS the title bar.
  WebkitAppRegion: 'drag',
  userSelect: 'none'
} as React.CSSProperties

const TITLE: React.CSSProperties = {
  color: 'var(--foreground, #eee)',
  fontSize: 12,
  fontWeight: 600,
  lineHeight: 1.2
}

const SUBTITLE: React.CSSProperties = {
  color: 'var(--muted-foreground, #8a8a8a)',
  fontSize: 11,
  lineHeight: 1.4,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap'
}

const CLOSE_BUTTON: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  borderRadius: 6,
  color: 'var(--muted-foreground, #8a8a8a)',
  cursor: 'pointer',
  flexShrink: 0,
  fontSize: 12,
  lineHeight: 1,
  padding: '4px 6px',
  WebkitAppRegion: 'no-drag'
} as React.CSSProperties

const THREAD: React.CSSProperties = {
  display: 'flex',
  flex: 1,
  flexDirection: 'column',
  gap: 14,
  minHeight: 0,
  overflowY: 'auto',
  padding: '12px 14px'
}

const EMPTY: React.CSSProperties = {
  color: 'var(--muted-foreground, #8a8a8a)',
  fontSize: 12,
  lineHeight: 1.5,
  margin: 0
}

const QUESTION: React.CSSProperties = {
  color: 'var(--foreground, #eee)',
  fontSize: 13,
  fontWeight: 600,
  lineHeight: 1.45,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word'
}

const ANSWER: React.CSSProperties = {
  color: 'var(--foreground, #eee)',
  fontSize: 13,
  lineHeight: 1.55,
  opacity: 0.92,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word'
}

const PENDING: React.CSSProperties = {
  color: 'var(--muted-foreground, #8a8a8a)',
  fontSize: 12,
  fontStyle: 'italic'
}

const ERROR: React.CSSProperties = {
  color: 'var(--destructive, #e5484d)',
  fontSize: 12,
  lineHeight: 1.5,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word'
}

const COMPOSER_ROW: React.CSSProperties = {
  borderTop: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.25))',
  flexShrink: 0,
  padding: '8px 10px 10px'
}

const COMPOSER: React.CSSProperties = {
  background: 'transparent',
  border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.3))',
  borderRadius: 8,
  color: 'var(--foreground, #eee)',
  fontFamily: 'inherit',
  fontSize: 13,
  lineHeight: 1.45,
  outline: 'none',
  padding: '7px 9px',
  resize: 'none',
  width: '100%'
}

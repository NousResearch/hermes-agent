import { useEffect, useReducer, useRef } from 'react'

import {
  initialQuickComposerState,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickComposerState
} from '@/store/quick-entry'

/**
 * The Quick Entry composer — the whole renderer surface of the global-hotkey
 * mini window. Deliberately one input plus a session-target picker and nothing
 * else: this is a capture surface, not a second chat.
 *
 * All behavior rides `quickComposerReducer` (pure, unit-tested): submit sends
 * the trimmed text + target through the shell and asks to hide; an empty submit
 * does neither so a stray Enter can't make the window vanish; Escape and losing
 * focus dismiss without sending; a dead gateway disables the input entirely
 * (the reducer refuses the send AND the input paints the reconnect hint).
 *
 * The window itself has no gateway connection. Its view of backend truth — is
 * the gateway up, which recent sessions exist — is pushed in by the primary
 * renderer through main (`onState`), and its text goes back the same road to
 * the primary renderer's normal prompt-submit path.
 */
export function QuickEntryApp() {
  const inputRef = useRef<HTMLInputElement>(null)

  // The reducer returns { send, state }; this wrapper performs the side effect
  // (hand the payload to the shell, ask to hide) and stores the next state, so
  // the decision stays pure and testable while the effects stay in one place.
  const [state, dispatch] = useReducer((current: QuickComposerState, event: QuickComposerEvent) => {
    const { send, state: next } = quickComposerReducer(current, event)
    const api = window.hermesDesktop?.quickEntry

    if (send) {
      api?.submit(send)
    } else if (!next.visible && current.visible) {
      api?.dismiss()
    }

    return next
  }, initialQuickComposerState)

  // Re-summoned by the chord: the shell reuses the window, so reset the draft
  // and take the keyboard back for a fresh capture. Also adopt gateway-state
  // pushes (connection + recent sessions) relayed from the primary renderer.
  useEffect(() => {
    const api = window.hermesDesktop?.quickEntry

    const offShown = api?.onShown(() => {
      dispatch({ type: 'shown' })
      requestAnimationFrame(() => inputRef.current?.focus())
    })

    const offState = api?.onState(payload => {
      // Echo the adopted payload back so main stops retrying its replay (see
      // electron/quick-entry-state-relay.ts); optional-chained because an
      // older preload may not expose it yet.
      api?.ackState?.(payload)
      dispatch({
        connected: payload?.connected === true,
        sessions: Array.isArray(payload?.sessions) ? payload.sessions : [],
        type: 'state'
      })
    })

    inputRef.current?.focus()

    return () => {
      offShown?.()
      offState?.()
    }
  }, [])

  return (
    <div
      style={{
        alignItems: 'center',
        background: 'transparent',
        display: 'flex',
        height: '100vh',
        justifyContent: 'center',
        padding: 12,
        width: '100vw'
      }}
    >
      <div
        style={{
          background: 'var(--ui-bg-elevated, var(--background))',
          border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.35))',
          borderRadius: 12,
          boxShadow: '0 18px 48px rgba(0,0,0,0.38)',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          padding: '10px 14px',
          width: '100%'
        }}
      >
        <div style={{ alignItems: 'center', display: 'flex', gap: 10 }}>
          <span
            aria-hidden
            style={{
              color: 'var(--muted-foreground, #8a8a8a)',
              flexShrink: 0,
              fontSize: 15,
              lineHeight: 1,
              userSelect: 'none'
            }}
          >
            ›
          </span>
          <input
            aria-label="Quick Entry"
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            disabled={!state.connected}
            onBlur={event => {
              // Moving focus to the target picker is not leaving the window.
              if (!event.relatedTarget) {
                dispatch({ type: 'blur' })
              }
            }}
            onChange={event => dispatch({ draft: event.target.value, type: 'edit' })}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                dispatch({ type: 'submit' })
              } else if (event.key === 'Escape') {
                event.preventDefault()
                dispatch({ type: 'dismiss' })
              }
            }}
            placeholder={state.connected ? 'Ask Hermes…' : 'Not connected — open Hermes to reconnect'}
            ref={inputRef}
            spellCheck={false}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--foreground, #eee)',
              flex: 1,
              fontFamily: 'inherit',
              fontSize: 15,
              minWidth: 0,
              opacity: state.connected ? 1 : 0.55,
              outline: 'none'
            }}
            value={state.draft}
          />
        </div>
        <div style={{ alignItems: 'center', display: 'flex', gap: 8 }}>
          <label
            htmlFor="quick-entry-target"
            style={{
              color: 'var(--muted-foreground, #8a8a8a)',
              flexShrink: 0,
              fontSize: 11,
              userSelect: 'none'
            }}
          >
            Send to
          </label>
          <select
            aria-label="Target session"
            disabled={!state.connected}
            id="quick-entry-target"
            onChange={event => dispatch({ target: event.target.value, type: 'target' })}
            onKeyDown={event => {
              if (event.key === 'Escape') {
                event.preventDefault()
                dispatch({ type: 'dismiss' })
              }
            }}
            style={{
              background: 'transparent',
              border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.35))',
              borderRadius: 6,
              color: 'var(--foreground, #eee)',
              fontSize: 11,
              maxWidth: 320,
              padding: '2px 6px'
            }}
            value={state.target}
          >
            <option value={QUICK_TARGET_CURRENT}>Current chat</option>
            <option value={QUICK_TARGET_NEW}>New session</option>
            {state.sessions.map(session => (
              <option key={session.id} value={session.id}>
                {session.title}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}

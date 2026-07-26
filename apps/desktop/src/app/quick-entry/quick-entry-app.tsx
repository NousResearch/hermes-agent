import { useEffect, useReducer, useRef } from 'react'

import {
  initialQuickComposerState,
  type QuickComposerEvent,
  quickComposerReducer,
  type QuickComposerState
} from '@/store/quick-entry'

/**
 * The Quick Entry composer — the whole renderer surface of the global-hotkey
 * mini window. Deliberately one input and nothing else: this is a capture
 * surface, not a second chat.
 *
 * All behavior rides `quickComposerReducer` (pure, unit-tested): submit sends
 * the trimmed text through the shell and asks to hide; an empty submit does
 * neither so a stray Enter can't make the window vanish; Escape and losing
 * focus dismiss without sending.
 *
 * The window itself has no gateway connection. Text goes to the main process,
 * which forwards it to the primary renderer's normal prompt-submit path.
 */
export function QuickEntryApp() {
  const inputRef = useRef<HTMLInputElement>(null)

  // The reducer returns { send, state }; this wrapper performs the side effect
  // (hand the text to the shell, ask to hide) and stores the next state, so the
  // decision stays pure and testable while the effects stay in one place.
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
  // and take the keyboard back for a fresh capture.
  useEffect(() => {
    const off = window.hermesDesktop?.quickEntry?.onShown(() => {
      dispatch({ type: 'shown' })
      requestAnimationFrame(() => inputRef.current?.focus())
    })

    inputRef.current?.focus()

    return off
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
          alignItems: 'center',
          background: 'var(--ui-bg-elevated, var(--background))',
          border: '1px solid var(--ui-stroke-secondary, rgba(127,127,127,0.35))',
          borderRadius: 12,
          boxShadow: '0 18px 48px rgba(0,0,0,0.38)',
          display: 'flex',
          gap: 10,
          padding: '10px 14px',
          width: '100%'
        }}
      >
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
          onBlur={() => dispatch({ type: 'blur' })}
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
          placeholder="Ask Hermes…"
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
            outline: 'none'
          }}
          value={state.draft}
        />
      </div>
    </div>
  )
}

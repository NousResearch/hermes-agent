import { useCallback, useEffect, useRef, useState } from 'react'

import type { HermesGateway } from '@/hermes'

/**
 * Ghost-text reply suggestion for the empty composer.
 *
 * When an agent turn completes (busy true → false) and the composer is empty,
 * ask the gateway's deterministic `complete.suggest` for a likely reply and
 * surface the first candidate as ghost text in the placeholder slot. Tab
 * accepts it into real text; typing or a new turn clears it; Esc dismisses it
 * until the next turn. Decorative by contract — any RPC failure resolves to
 * "no ghost", never an error, and never blocks the composer.
 *
 * Backend note: `complete.suggest` is the same gateway RPC the Ink TUI uses;
 * `gateway.request` returns the unwrapped result (matching `useSlashCompletions`).
 */

export interface GhostCandidate {
  kind: string
  text: string
}

interface SuggestResult {
  candidates?: GhostCandidate[]
  history_version?: number
}

/** First candidate's text from a `complete.suggest` result, '' when none. */
export function firstGhostCandidate(result: unknown): string {
  const text = (result as SuggestResult | null | undefined)?.candidates?.[0]?.text

  return typeof text === 'string' ? text.trim() : ''
}

/** Whether the ghost should render: only in an idle, empty, undismissed
 *  composer that actually has a suggestion. Pure so the contract is testable. */
export function ghostVisible(o: { busy: boolean; dismissed: boolean; empty: boolean; ghost: string }): boolean {
  return o.empty && !o.busy && !o.dismissed && o.ghost !== ''
}

/** Plain Tab only: a modified Tab keeps its native meaning. */
const isPlainTab = (e: KeyboardEvent) => e.key === 'Tab' && !e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey

export function useComposerGhost(opts: {
  busy: boolean
  gateway: HermesGateway | null | undefined
  hasText: boolean
  placeholder: string
  /** Accepted ghost text goes into the composer (never sent), then focus returns to it. */
  requestMainFocus: () => void
  sessionId: null | string | undefined
  setComposerText: (text: string) => void
  /** An open `/` or `@` drawer owns Tab/Escape. */
  trigger: unknown
}): {
  acceptGhost: () => null | string
  dismissGhost: () => boolean
  ghost: string
  /** Tab accepts / Esc dismisses a visible ghost; true when the key was consumed. */
  handleGhostKeyDown: (event: { key: string; nativeEvent: KeyboardEvent; preventDefault: () => void }) => boolean
  /** The ghost while visible, else the regular placeholder. */
  placeholder: string
} {
  const { busy, gateway, hasText, placeholder, requestMainFocus, sessionId, setComposerText, trigger } = opts
  const empty = !hasText
  const [ghost, setGhost] = useState('')
  const [dismissed, setDismissed] = useState(false)
  const prevBusyRef = useRef(busy)
  const seqRef = useRef(0)

  // Fire exactly once per completed turn: on the busy → idle transition.
  // eslint-disable-next-line no-restricted-syntax -- this records a prop edge and request token, not an atom mirror
  useEffect(() => {
    const wasBusy = prevBusyRef.current
    prevBusyRef.current = busy

    if (!wasBusy || busy) {
      return
    }

    if (!gateway || !sessionId || !empty) {
      return
    }

    const seq = ++seqRef.current

    setDismissed(false)
    setGhost('')

    gateway
      .request<SuggestResult>('complete.suggest', { session_id: sessionId })
      .then(result => {
        if (seq === seqRef.current) {
          setGhost(firstGhostCandidate(result))
        }
      })
      .catch(() => {
        // Decorative: a failed suggestion must never reach the composer.
      })
  }, [busy, empty, gateway, sessionId])

  // Typing (empty → false) clears the ghost so it never overlaps real text.
  useEffect(() => {
    if (!empty) {
      setGhost('')
      setDismissed(false)
    }
  }, [empty])

  // A different conversation invalidates any in-flight or shown suggestion.
  // eslint-disable-next-line no-restricted-syntax -- this invalidates a request token and does not mirror reactive state
  useEffect(() => {
    seqRef.current += 1
    setGhost('')
    setDismissed(false)
  }, [sessionId])

  const visible = ghostVisible({ busy, dismissed, empty, ghost })

  const acceptGhost = useCallback((): null | string => {
    if (!visible) {
      return null
    }

    setGhost('')

    return ghost
  }, [ghost, visible])

  const dismissGhost = useCallback((): boolean => {
    if (!visible) {
      return false
    }

    setDismissed(true)

    return true
  }, [visible])

  const handleGhostKeyDown = useCallback(
    (event: { key: string; nativeEvent: KeyboardEvent; preventDefault: () => void }): boolean => {
      if (trigger || !visible) {
        return false
      }

      if (isPlainTab(event.nativeEvent)) {
        // preventDefault keeps Tab from moving focus to the model picker.
        event.preventDefault()
        setGhost('')
        setComposerText(ghost)
        requestMainFocus()

        return true
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        setDismissed(true)

        return true
      }

      return false
    },
    [ghost, requestMainFocus, setComposerText, trigger, visible]
  )

  return {
    acceptGhost,
    dismissGhost,
    ghost: visible ? ghost : '',
    handleGhostKeyDown,
    placeholder: visible ? ghost : placeholder
  }
}

import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resetBrowseState } from '@/store/composer-input-history'

interface UseComposerPlaceholderOptions {
  disabled: boolean
  reconnecting: boolean
  sessionId: null | string | undefined
}

type PlaceholderKind = 'followUp' | 'newSession'

interface PlaceholderChoice {
  index: number
  kind: PlaceholderKind
}

function choosePlaceholder(kind: PlaceholderKind, pool: readonly string[]): PlaceholderChoice {
  return { index: pool.length === 0 ? 0 : Math.floor(Math.random() * pool.length), kind }
}

function resolvePlaceholder(choice: PlaceholderChoice, pool: readonly string[]): string {
  if (pool.length === 0) {
    return ''
  }

  return pool[choice.index % pool.length] ?? pool[0] ?? ''
}

/**
 * Keep a language-independent starter/follow-up selection for the active
 * conversation. Locale changes only change the text at that selection; a new
 * token is chosen only for a genuine conversation change. A null-to-id persist
 * keeps the current starter and never resets input-history state.
 */
export function useComposerPlaceholder({ disabled, reconnecting, sessionId }: UseComposerPlaceholderOptions): string {
  const { t } = useI18n()
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders

  const [choice, setChoice] = useState<PlaceholderChoice>(() => {
    const kind = sessionId ? 'followUp' : 'newSession'
    const pool = kind === 'followUp' ? followUpPlaceholders : newSessionPlaceholders

    return choosePlaceholder(kind, pool)
  })

  const prevSessionIdRef = useRef(sessionId)

  useEffect(() => {
    const prev = prevSessionIdRef.current
    prevSessionIdRef.current = sessionId

    if (prev === sessionId || (prev == null && sessionId == null)) {
      return
    }

    if (prev == null && sessionId) {
      return
    }

    resetBrowseState(prev)
    const kind = sessionId ? 'followUp' : 'newSession'
    const pool = kind === 'followUp' ? followUpPlaceholders : newSessionPlaceholders

    setChoice(choosePlaceholder(kind, pool))
  }, [followUpPlaceholders, newSessionPlaceholders, sessionId])

  const restingPool = choice.kind === 'followUp' ? followUpPlaceholders : newSessionPlaceholders
  const restingPlaceholder = resolvePlaceholder(choice, restingPool)

  // When the transport is disabled it's because the gateway isn't open.
  // Distinguish a cold start ("Starting Hermes...") from a dropped connection
  // we're trying to restore. During reconnect, keep the textbox editable so a
  // flaky network doesn't block drafting; only submit/backend actions stay
  // disabled until the gateway is open again.
  return disabled
    ? reconnecting
      ? t.composer.placeholderReconnecting
      : t.composer.placeholderStarting
    : restingPlaceholder
}

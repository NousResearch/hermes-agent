import { useEffect, useRef, useState } from 'react'

import type { Locale } from '@/i18n'

export const pickComposerPlaceholder = (pool: readonly string[]) => pool[Math.floor(Math.random() * pool.length)]

export interface UseRestingComposerPlaceholderOptions {
  followUpPlaceholders: readonly string[]
  locale: Locale
  newSessionPlaceholders: readonly string[]
  onConversationChanged?: (previousSessionId: string | null | undefined) => void
  sessionId?: string | null
}

export function useRestingComposerPlaceholder({
  followUpPlaceholders,
  locale,
  newSessionPlaceholders,
  onConversationChanged,
  sessionId
}: UseRestingComposerPlaceholderOptions): string {
  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickComposerPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders)
  )

  const previousRef = useRef<{ locale: Locale; sessionId: string | null | undefined }>({ locale, sessionId })

  useEffect(() => {
    const previous = previousRef.current
    const localeChanged = previous.locale !== locale
    const sessionChanged = previous.sessionId !== sessionId

    previousRef.current = { locale, sessionId }

    if (!localeChanged && !sessionChanged) {
      return
    }

    // null -> id means the brand-new session was just persisted. Keep the
    // starter placeholder stable unless the visible language also changed.
    if (!localeChanged && previous.sessionId == null && sessionId) {
      return
    }

    if (sessionChanged) {
      onConversationChanged?.(previous.sessionId)
    }

    setRestingPlaceholder(pickComposerPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders))
  }, [followUpPlaceholders, locale, newSessionPlaceholders, onConversationChanged, sessionId])

  return restingPlaceholder
}

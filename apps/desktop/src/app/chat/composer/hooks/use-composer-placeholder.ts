import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resetBrowseState } from '@/store/composer-input-history'

import { pickPlaceholder } from '../composer-utils'

interface UseComposerPlaceholderOptions {
  disabled: boolean
  reconnecting: boolean
  sessionId: null | string | undefined
}

/**
 * The composer's placeholder text. A resting starter (new session) / continuation
 * (existing session) is picked once and only re-rolled when we genuinely move to
 * a *different* conversation — the null→id persist of a freshly-started session
 * keeps its starter so the text doesn't flip mid-stream. While the transport is
 * down, it swaps to a reconnecting / starting message instead.
 */
export function useComposerPlaceholder({ disabled, reconnecting, sessionId }: UseComposerPlaceholderOptions): string {
  const { locale, t } = useI18n()
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders

  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders)
  )

  const previousRef = useRef({ locale, sessionId })

  useEffect(() => {
    const previous = previousRef.current
    const localeChanged = previous.locale !== locale
    const sessionChanged = previous.sessionId !== sessionId
    const initialSessionPersisted = sessionChanged && previous.sessionId == null && Boolean(sessionId)

    previousRef.current = { locale, sessionId }

    if (!localeChanged && !sessionChanged) {
      return
    }

    // null → id: the new session we're already in just got persisted. Keep the
    // starter we showed instead of swapping to a follow-up under the user. A
    // simultaneous locale change may translate it, but must not reset history.
    if (sessionChanged && !initialSessionPersisted) {
      resetBrowseState(previous.sessionId)
    }

    if (localeChanged || !initialSessionPersisted) {
      setRestingPlaceholder(pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders))
    }
  }, [followUpPlaceholders, locale, newSessionPlaceholders, sessionId])

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

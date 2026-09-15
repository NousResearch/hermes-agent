import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resetBrowseState } from '@/store/composer-input-history'
import { $composerSendPrefs } from '@/store/composer-send'

import { pickPlaceholder } from '../composer-utils'
import { composerSendModeHint } from '../send-mode-hint'

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
 *
 * When the send mode isn't the default, the resting text names the live gesture
 * (`Enter starts a new line · tap Enter twice to send`). An empty composer is
 * where a misfire is impossible and the gesture is easiest to misremember, and
 * the hint doubles as "your custom mode is on" — which a user otherwise has no
 * ambient way to see. The default mode appends nothing: telling everyone what
 * Enter already does is noise.
 */
export function useComposerPlaceholder({ disabled, reconnecting, sessionId }: UseComposerPlaceholderOptions): string {
  const { t } = useI18n()
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders
  const { mode } = useStore($composerSendPrefs)

  const sendHint =
    mode === 'enter'
      ? null
      : composerSendModeHint(mode, {
          chord: t.composer.placeholderSendChord,
          doubleTap: t.composer.placeholderSendDoubleTap,
          enterSends: t.composer.placeholderSendEnterSends,
          newline: t.composer.placeholderSendNewline,
          pause: t.composer.placeholderSendPause
        })

  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders)
  )

  const prevSessionIdRef = useRef(sessionId)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const prev = prevSessionIdRef.current
    prevSessionIdRef.current = sessionId

    if (prev === sessionId) {
      return
    }

    // null → id: the new session we're already in just got persisted. Keep the
    // starter we showed instead of swapping to a follow-up under the user.
    if (prev == null && sessionId) {
      return
    }

    resetBrowseState(prev)
    setRestingPlaceholder(pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders))
  }, [followUpPlaceholders, newSessionPlaceholders, sessionId])

  // When the transport is disabled it's because the gateway isn't open.
  // Distinguish a cold start ("Starting Hermes...") from a dropped connection
  // we're trying to restore. During reconnect, keep the textbox editable so a
  // flaky network doesn't block drafting; only submit/backend actions stay
  // disabled until the gateway is open again.
  if (disabled) {
    return reconnecting ? t.composer.placeholderReconnecting : t.composer.placeholderStarting
  }

  return sendHint ? `${restingPlaceholder} · ${sendHint}` : restingPlaceholder
}

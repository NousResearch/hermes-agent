import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { $backendTransitions, backendTransitionLabel, transitionProfileKey } from '@/store/backend-transition'
import { resetBrowseState } from '@/store/composer-input-history'
import { $activeGatewayProfile } from '@/store/profile'

import { pickPlaceholder } from '../composer-utils'

interface UseComposerPlaceholderOptions {
  disabled: boolean
  /** The composer's profile; the active gateway profile when absent. */
  profile?: null | string
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
export function useComposerPlaceholder({
  disabled,
  profile,
  reconnecting,
  sessionId
}: UseComposerPlaceholderOptions): string {
  const { t } = useI18n()
  const activeProfile = useStore($activeGatewayProfile)
  const transition = useStore($backendTransitions)[transitionProfileKey(profile || activeProfile)] ?? null
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders

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
  // disabled until the gateway is open again. A plugin-run profile's planned
  // start, restart or stop names itself instead, also while the connection is
  // still open (a start: this computer's backend serves the profile meanwhile).
  return transition
    ? backendTransitionLabel(transition, t.sandbox)
    : disabled
      ? reconnecting
        ? t.composer.placeholderReconnecting
        : t.composer.placeholderStarting
      : restingPlaceholder
}

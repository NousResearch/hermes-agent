import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { isFreshDraftScope } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'

import { pickPlaceholder } from '../composer-utils'

interface UseComposerPlaceholderOptions {
  disabled: boolean
  reconnecting: boolean
  /** The composer's durable draft scope: a stored session key, or a fresh draft's. */
  scope: null | string | undefined
  /** Runtime session id; input-history browsing is keyed on it. */
  sessionId: null | string | undefined
}

const existingConversation = (scope: null | string | undefined) => Boolean(scope) && !isFreshDraftScope(scope)

/**
 * The composer's placeholder text. A resting starter (new chat) / continuation
 * (existing chat) is picked once and only re-rolled when we genuinely move to a
 * *different* conversation.
 *
 * Identity is the durable draft scope, not the runtime id: a routed chat owns
 * its stored key the moment the route lands, while its runtime may bind much
 * later (paint-first resume), and compression rotates the runtime under the
 * same chat. A fresh draft becoming a stored chat is either this composer's
 * first send getting its session (keep the starter, so the text doesn't flip
 * mid-stream) or the user opening an existing chat from the landing (re-roll);
 * only the send's create handoff, reporting the scope it created through
 * `sessionCreated`, tells them apart. While the transport is down, it swaps to a reconnecting / starting
 * message instead.
 */
export function useComposerPlaceholder({ disabled, reconnecting, scope, sessionId }: UseComposerPlaceholderOptions): {
  placeholder: string
  sessionCreated: (scope: string) => void
} {
  const { t } = useI18n()
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders

  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickPlaceholder(existingConversation(scope) ? followUpPlaceholders : newSessionPlaceholders)
  )

  const prevScopeRef = useRef(scope)
  const prevSessionIdRef = useRef(sessionId)
  const createdScopeRef = useRef<null | string>(null)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const prev = prevSessionIdRef.current
    prevSessionIdRef.current = sessionId

    if (prev !== sessionId) {
      resetBrowseState(prev)
    }
  }, [sessionId])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const prev = prevScopeRef.current
    prevScopeRef.current = scope

    // Fresh-draft keys come and go without the conversation changing.
    if (prev === scope || (!existingConversation(prev) && !existingConversation(scope))) {
      return
    }

    // The new chat we're already in just got persisted. Keep the starter we
    // showed instead of swapping to a follow-up under the user.
    if (scope === createdScopeRef.current) {
      return
    }

    createdScopeRef.current = null

    setRestingPlaceholder(pickPlaceholder(existingConversation(scope) ? followUpPlaceholders : newSessionPlaceholders))
  }, [followUpPlaceholders, newSessionPlaceholders, scope])

  const placeholder = disabled
    ? reconnecting
      ? t.composer.placeholderReconnecting
      : t.composer.placeholderStarting
    : restingPlaceholder

  return { placeholder, sessionCreated: created => (createdScopeRef.current = created) }
}

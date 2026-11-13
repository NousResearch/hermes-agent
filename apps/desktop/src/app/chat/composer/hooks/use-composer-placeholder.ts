import { useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { isFreshDraftScope } from '@/store/composer'
import { resetBrowseState } from '@/store/composer-input-history'
import { usePortalShell } from '@/store/shell-variant'

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

type Identity = null | string | undefined

/** Whether the composer sits in an existing conversation (follow-up text) or a new one (starter). */
const inConversation = (identity: Identity, redesign: boolean) =>
  redesign ? existingConversation(identity) : Boolean(identity)

/** Whether moving from `prev` to `next` is a different conversation, which re-rolls the text.
 *  Classic keeps main's rule, keyed on the runtime id: a session opened paint-first keeps the
 *  starter it showed while its runtime was unbound, since null → id reads as "the new session we're
 *  in just got persisted". The redesign keys on the durable scope (below). */
function movedConversation(prev: Identity, next: Identity, created: null | string, redesign: boolean): boolean {
  if (prev === next) {
    return false
  }

  if (!redesign) {
    return !(prev == null && next)
  }

  // Fresh-draft keys come and go without the conversation changing, and the new chat we're already
  // in just got persisted when its own send reports the scope it created.
  return (existingConversation(prev) || existingConversation(next)) && next !== created
}

/**
 * The composer's placeholder text. A resting starter (new chat) / continuation
 * (existing chat) is picked once and only re-rolled when we genuinely move to a
 * *different* conversation.
 *
 * In the redesign, identity is the durable draft scope, not the runtime id (classic keeps main's
 * runtime-id rule, see movedConversation): a routed chat owns
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
  const redesign = usePortalShell()
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders
  const identity = redesign ? scope : sessionId

  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickPlaceholder(inConversation(identity, redesign) ? followUpPlaceholders : newSessionPlaceholders)
  )

  const prevIdentityRef = useRef(identity)
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
    const prev = prevIdentityRef.current
    prevIdentityRef.current = identity

    if (!movedConversation(prev, identity, createdScopeRef.current, redesign)) {
      return
    }

    createdScopeRef.current = null

    setRestingPlaceholder(
      pickPlaceholder(inConversation(identity, redesign) ? followUpPlaceholders : newSessionPlaceholders)
    )
  }, [followUpPlaceholders, newSessionPlaceholders, identity, redesign])

  const placeholder = disabled
    ? reconnecting
      ? t.composer.placeholderReconnecting
      : t.composer.placeholderStarting
    : restingPlaceholder

  return { placeholder, sessionCreated: created => (createdScopeRef.current = created) }
}

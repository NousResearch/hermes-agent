import { useStore } from '@nanostores/react'
import { useEffect, useId, useState } from 'react'

import { useStoreSelector } from '@/lib/use-session-slice'
import { $billingBlock } from '@/store/billing-block'
import { $collapsedStatusDrawers, setStatusDrawerCollapsed, statusDrawerKey } from '@/store/composer-status-drawer'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { isSessionOwnerRoute } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

import { useComposerScope } from '../scope'

/** `statusSessionId`: the chat the status stack speaks for (an out-of-credits wall there opens it). */
export function useStatusDrawer(sessionKey: string | null, statusSessionId?: string | null) {
  const scope = useComposerScope()
  const activeProfile = useStore($activeGatewayProfile)
  const activeConnection = useStore($activeConnectionId)
  const owner = knownOwnerForSession(sessionKey)
  const route = isSessionOwnerRoute(owner) ? owner : null

  const profile = normalizeProfileKey(
    route?.profile ?? (typeof owner === 'string' ? owner : (scope.profile ?? activeProfile))
  )

  const targetProfile = normalizeProfileKey(route?.targetProfile ?? profile)
  const connectionId = route ? (route.connectionId ?? null) : owner ? null : (scope.connectionId ?? activeConnection)
  const ownerKey = JSON.stringify([connectionId, profile, targetProfile])
  const key = sessionKey ? statusDrawerKey({ connectionId, profile, targetProfile, sessionId: sessionKey }) : null
  const saved = useStoreSelector($collapsedStatusDrawers, keys => key !== null && keys.includes(key))
  const draftKey = key ?? ownerKey
  const [draft, setDraft] = useState({ key: draftKey, collapsed: false })

  // Unsaved drafts have no durable preference. Never carry their choice into
  // a stored conversation selected from the sidebar or a subsequent new chat.
  if (draft.key !== draftKey) {
    setDraft({ key: draftKey, collapsed: false })
  }

  const collapsed = key ? saved : draft.key === draftKey && draft.collapsed

  const toggle = () => {
    if (key) {
      setStatusDrawerCollapsed(key, !collapsed)
    } else {
      setDraft({ key: draftKey, collapsed: !collapsed })
    }
  }

  // An out-of-credits wall for this chat is the drawer's banner (the turn's card defers to it).
  const billingWallAt = useStoreSelector($billingBlock, block =>
    statusSessionId && block?.sessionId === statusSessionId ? block.at : null
  )

  useRevealStatusDrawer(billingWallAt, collapsed, toggle)

  // Stable ids for the two drawer regions (status stack, coding row) their toggles control.
  const statusDrawerId = useId()
  const codingDrawerId = useId()

  return { codingDrawerId, collapsed, statusDrawerId, toggle }
}

/** Open a collapsed drawer once per new `token` (e.g. an out-of-credits wall for this chat):
 *  the drawer is that alert's one surface, so it must not arrive hidden. A later collapse
 *  by the user sticks until the next token. */
export function useRevealStatusDrawer(token: number | null, collapsed: boolean, toggle: () => void) {
  const [revealed, setRevealed] = useState<number | null>(null)

  useEffect(() => {
    if (token === null || revealed === token) {
      return
    }

    setRevealed(token)

    if (collapsed) {
      toggle()
    }
  }, [collapsed, revealed, toggle, token])
}

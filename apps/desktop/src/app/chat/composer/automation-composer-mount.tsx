import { useStore } from '@nanostores/react'

import { NEW_SESSION_TITLE, sessionTitle } from '@/lib/chat-runtime'
import { $automationComposer } from '@/store/automation-composer'
import { openRouteTile } from '@/store/route-tiles'
import { $sessions, sessionMatchesStoredId } from '@/store/session'

import { AutomationComposerDialog } from './automation-composer-dialog'
import type { OnSubmitText } from './automation-composer-form'

interface AutomationComposerMountProps {
  /** Submits a goal kickoff message to the captured session (the composer's own submit). */
  onSubmitText: OnSubmitText
  /** The session this composer owns. Several composers can be mounted at once (route tiles, bot
   *  chats), so the dialog renders only inside the one that opened it — never in every composer
   *  sharing the global open state. */
  sessionId?: string | null
}

/**
 * Renders the automation dialog for the conversation this composer owns, or nothing.
 *
 * Kept out of `ChatBar` on purpose: it owns its own store reads (open state, the
 * conversation's title, the Cron route), so the composer body does not grow a
 * branch per automation affordance.
 */
export function AutomationComposerMount({ onSubmitText, sessionId }: AutomationComposerMountProps) {
  const state = useStore($automationComposer)
  const sessions = useStore($sessions)
  const ownerId = sessionId ?? null

  if (!state.open || !state.sessionId || state.sessionId !== ownerId) {
    return null
  }

  // The title belongs to the session that opened the dialog, not to whichever
  // session the sidebar currently calls active.
  const owner = sessions.find(session => sessionMatchesStoredId(session, ownerId))

  return (
    <AutomationComposerDialog
      conversationTitle={owner ? sessionTitle(owner) : NEW_SESSION_TITLE}
      onOpenCron={() => openRouteTile('/cron')}
      onSubmitText={onSubmitText}
    />
  )
}

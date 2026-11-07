import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import { useI18n } from '@/i18n'
import { $compactingSessions } from '@/store/compaction'
import { confirm } from '@/store/confirm'
import { notifyError } from '@/store/notifications'
import { $activeSessionId, $busy, $sessions } from '@/store/session'
import { $focusedRuntimeId, $sessionStates, sessionTileDelegate } from '@/store/session-states'

export interface CompactConversationAction {
  /** A turn is running: the gateway refuses compression until it ends. */
  busy: boolean
  /** The focused chat is already compacting (this button, `/compress` or auto). */
  pending: boolean
  /** No runtime session to compact yet (fresh draft). */
  unavailable: boolean
  onRequest: () => void
}

/** The chat's own title, so the confirm names what compacts when another
 *  surface (a Bot group room) is in front of it. Untitled chats keep the
 *  generic copy rather than a preview snippet. */
function chatTitle(runtimeId: string): string | undefined {
  const storedId = $sessionStates.get()[runtimeId]?.storedSessionId

  return storedId
    ? $sessions
        .get()
        .find(session => session.id === storedId)
        ?.title?.trim() || undefined
    : undefined
}

function sessionBusy(sessionId: string): boolean {
  return Boolean($sessionStates.get()[sessionId]?.busy) || ($activeSessionId.get() === sessionId && $busy.get())
}

/** The context popover's "Compact conversation" button.
 *
 *  The work is the typed `/compress` pipeline (`slash.ts` `compress`), reached
 *  through the session delegate the same way the overflow card's button does,
 *  so owner routing, stale-runtime recovery, transcript replacement, progress
 *  notices and the in-flight coalesce are shared rather than re-implemented.
 *
 *  The chat is captured at click time and re-checked after the confirmation
 *  resolves: the user can switch chats or start a reply while the dialog is
 *  open, and the action must then do nothing rather than compact whichever
 *  chat is on screen by then. */
export function useCompactConversation({
  busy,
  sessionId
}: {
  busy: boolean
  sessionId: null | string
}): CompactConversationAction {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel.compact
  const failed = t.assistant.thread.errorCompressFailed
  const compacting = useStore($compactingSessions)
  const pending = Boolean(sessionId && compacting[sessionId])

  const onRequest = useCallback(() => {
    const target = sessionId

    if (!target || busy || pending) {
      return
    }

    void (async () => {
      const title = chatTitle(target)

      const ok = await confirm({
        confirmLabel: copy.confirm,
        description: copy.hint,
        title: title ? copy.questionNamed(title) : copy.question
      })

      if (!ok || $focusedRuntimeId.get() !== target || sessionBusy(target) || $compactingSessions.get()[target]) {
        return
      }

      await sessionTileDelegate()?.executeSlash('/compress', target, { typed: false })
    })().catch(error => notifyError(error, failed))
  }, [busy, copy, failed, pending, sessionId])

  return useMemo(() => ({ busy, onRequest, pending, unavailable: !sessionId }), [busy, onRequest, pending, sessionId])
}

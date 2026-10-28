import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router'

import { useTemporaryChatButterbar } from '@/app/chat/temporary/temporary-chat-butterbar'
import { openSession } from '@/app/open-session'
import { $workspaceIsPage } from '@/app/routes'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { $newChatProfile } from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'
import {
  $newTemporaryChatRequest,
  $temporaryChatEndRequests,
  $temporaryChats,
  expiredTemporaryChats,
  forgetTemporaryChat,
  takeTemporaryChatEndRequests,
  TEMPORARY_CHAT_IDLE_TTL_MS,
  temporaryChatOnScreen,
  updateTemporaryIdleClocks
} from '@/store/temporary-chat'

import { temporaryChatBusy } from './temporary-chat-busy'

const IDLE_TICK_MS = 15_000

/**
 * Owns the life of temporary chats in this window:
 * - "New temporary chat" requests → a fresh temporary draft (the same draft as New Chat).
 * - Leaving a temporary chat does NOT delete it (a reply may still be streaming; the user must be
 *   able to come back and stop it). Its idle clock starts once it is off screen with no turn running.
 * - It is deleted when the user ends it (End / ⌘W) or its idle clock passes the TTL. Deletion is
 *   `removeSession` — the same teardown as Delete: interrupt, `session.close` (a hard delete for a
 *   temporary chat), REST delete as the backstop, local cache purge.
 * Quitting the app is covered by the backend's shutdown teardown and its startup sweep.
 */
export function useTemporaryChatLifecycle(
  removeSession: (storedSessionId: string) => Promise<void>,
  startFreshSessionDraft: (options: { temporary: boolean }) => void,
  ttlMs = TEMPORARY_CHAT_IDLE_TTL_MS
): void {
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const workspaceIsPage = useStore($workspaceIsPage)
  const newTemporaryChatRequest = useStore($newTemporaryChatRequest)
  const endRequests = useStore($temporaryChatEndRequests)
  const lastRequestRef = useRef(newTemporaryChatRequest)
  const navigate = useNavigate()
  // "Back to temporary chat": the one open-a-session door, onto main (it is the user's chat).
  const openStoredSession = useCallback((storedId: string) => openSession(storedId, navigate, 'main'), [navigate])

  useTemporaryChatButterbar(openStoredSession)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write: consume each request once
  useEffect(() => {
    if (newTemporaryChatRequest === lastRequestRef.current) {
      return
    }

    lastRequestRef.current = newTemporaryChatRequest
    setWorkspaceScope('sessions')
    $newChatProfile.set(null)
    startFreshSessionDraft({ temporary: true })
  }, [newTemporaryChatRequest, startFreshSessionDraft])

  useEffect(() => {
    for (const storedId of takeTemporaryChatEndRequests()) {
      forgetTemporaryChat(storedId)
      void removeSession(storedId)
    }
  }, [endRequests, removeSession])

  useEffect(() => {
    const tick = () => {
      const now = Date.now()
      const onScreen = temporaryChatOnScreen($selectedStoredSessionId.get(), $workspaceIsPage.get())

      const active = (chat: { storedId: string; runtimeId: null | string }) =>
        chat.storedId === onScreen || temporaryChatBusy(chat.runtimeId)

      $temporaryChats.set(updateTemporaryIdleClocks($temporaryChats.get(), active, now))

      for (const storedId of expiredTemporaryChats($temporaryChats.get(), now, ttlMs)) {
        forgetTemporaryChat(storedId)
        void removeSession(storedId)
      }
    }

    tick()
    const timer = window.setInterval(tick, IDLE_TICK_MS)

    return () => window.clearInterval(timer)
  }, [removeSession, selectedStoredSessionId, workspaceIsPage, ttlMs])
}

import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $workspaceIsPage } from '@/app/routes'
import { useI18n } from '@/i18n'
import { Incognito } from '@/lib/icons'
import { registerButterbar } from '@/store/butterbar'
import { $selectedStoredSessionId } from '@/store/session'
import { $temporaryChats, $temporaryDraft } from '@/store/temporary-chat'

import { $runningTemporaryChats } from '../../session/hooks/temporary-chat-busy'

import { confirmEndTemporaryChat } from './confirm-end'

const BAR_ID = 'temporary-chat'
// A violet tint reads as "private", distinct from every status tone the bar already uses.
const PRIVATE_TINT = '#8b5cf6'

/**
 * The always-visible mark of private mode, in the app-wide butterbar:
 * - on screen (a temporary draft or chat): "Temporary chat — not saved …" (+ End for a live chat);
 * - elsewhere while a temporary chat is alive: the same bar with Back / End, so a reply that is
 *   still streaming is never out of reach.
 */
export function useTemporaryChatButterbar(openStoredSession: (storedSessionId: string) => void): void {
  const { t } = useI18n()
  const c = t.temporaryChat
  const chats = useStore($temporaryChats)
  const draft = useStore($temporaryDraft)
  const selected = useStore($selectedStoredSessionId)
  const workspaceIsPage = useStore($workspaceIsPage)
  const running = useStore($runningTemporaryChats)

  const onScreen = selected && !workspaceIsPage ? chats[selected] : undefined
  const chat = onScreen ?? (selected && chats[selected]) ?? Object.values(chats)[0]
  const away = Boolean(chat) && !onScreen
  const draftOnScreen = !selected && !workspaceIsPage && draft

  useEffect(() => {
    if (!chat && !draftOnScreen) {
      return
    }

    const busy = Boolean(chat && running.split('|').includes(chat.storedId))

    const node = (
      <span className="inline-flex min-w-0 items-center gap-2" data-testid="temporary-chat-bar">
        <strong className="font-medium">{c.title}</strong>
        <span className="truncate opacity-80">{busy ? c.running : c.badgeHint}</span>
        {chat && away && (
          <button
            className="underline underline-offset-2"
            onClick={() => openStoredSession(chat.storedId)}
            type="button"
          >
            {c.returnTo}
          </button>
        )}
        {chat && (
          <button
            className="underline underline-offset-2"
            onClick={() => void confirmEndTemporaryChat(chat.storedId, c)}
            type="button"
          >
            {c.endChat}
          </button>
        )}
      </span>
    )

    return registerButterbar({ color: PRIVATE_TINT, icon: <Incognito />, id: BAR_ID, node, priority: 100 })
  }, [away, c, chat, draftOnScreen, openStoredSession, running])
}

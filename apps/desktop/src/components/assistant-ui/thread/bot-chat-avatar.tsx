/**
 * The bot's face beside its own messages in a Bot Mode chat.
 *
 * A chat with a bot should read like a chat DM — the one you are talking to has
 * a face, the same way every room message already carries its speaker's. The
 * thread itself stays avatar-free for working sessions: the face is gated on the
 * live session being a bot chat (`isBotChatSession`), and resolves the profile's
 * avatar through the same gateway asset the inter-agent notices use.
 */
import { useStore } from '@nanostores/react'
import { type FC, useEffect, useState } from 'react'

import { $botChatScopes, isBotChatSession, storedSessionIdForRuntimeId } from '@/store/session-states'

import { agentAvatarCache, resolveAgentAvatar } from './user-message'

/** `bot:<handle>` owner key → the profile handle. Local owners carry a
 *  connection prefix (`bot:local::researcher`) that is not part of the name. */
export function botChatHandle(ownerKey: null | string | undefined): null | string {
  if (!ownerKey?.startsWith('bot:')) {
    return null
  }

  const handle = ownerKey.slice('bot:'.length).replace(/^local::/, '').trim()

  return handle || null
}

/** The handle of the bot whose chat this live session is, else null. */
export function useBotChatHandle(runtimeId: null | string | undefined): null | string {
  const scopes = useStore($botChatScopes)

  if (!runtimeId || !isBotChatSession(runtimeId)) {
    return null
  }

  const stored = storedSessionIdForRuntimeId(runtimeId)

  return botChatHandle(stored ? scopes[stored]?.workspaceOwnerKey : null)
}

/** The face itself: the profile's avatar, else the neutral agent glyph — the
 *  same fallback the inter-agent notices show, so an art-less bot still reads
 *  as an agent rather than a broken image. */
export const BotChatAvatar: FC<{ handle: string }> = ({ handle }) => {
  const [avatar, setAvatar] = useState<null | string>(() => agentAvatarCache.get(handle.toLowerCase()) ?? null)

  useEffect(() => {
    let live = true

    void resolveAgentAvatar(handle).then(url => {
      if (live && url) {
        setAvatar(url)
      }
    })

    return () => {
      live = false
    }
  }, [handle])

  if (avatar) {
    return (
      <img
        alt=""
        aria-hidden
        className="size-6 shrink-0 rounded-full object-cover"
        data-slot="bot-chat-avatar"
        src={avatar}
      />
    )
  }

  return (
    <span
      aria-hidden
      className="grid size-6 shrink-0 place-items-center rounded-full bg-(--chrome-action-hover) text-[0.7rem] leading-none"
      data-slot="bot-chat-avatar"
    >
      🤖
    </span>
  )
}

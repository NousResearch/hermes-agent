import type { FC } from 'react'
import { useMemo } from 'react'

import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { CHAT_EMPTY_AREA, type ChatEmptyContribution } from '@/lib/chat-empty'

/**
 * The empty transcript's contributed slot. Mounts the first registration and
 * lets it decide — it renders the session's empty state, or nothing at all if
 * the session isn't one it owns.
 *
 * Deliberately a mount rather than a claim the transcript resolves up front:
 * whether a session has an empty state depends on data the plugin loads on its
 * own clock (a bot chat's roster lands after the transcript), and only a
 * mounted component can subscribe and appear when it arrives.
 */
export const ChatEmptySlot: FC<{ sessionId: string }> = ({ sessionId }) => {
  const contributions = useContributions(CHAT_EMPTY_AREA)
  const match = contributions[0]
  const render = (match?.data as ChatEmptyContribution | undefined)?.render

  // Stable component identity: ContribRender mounts this AS a component, so a
  // fresh closure per render would remount the empty state on every tick.
  const renderEmpty = useMemo(() => (render ? () => render({ sessionId }) : null), [render, sessionId])

  if (!match || !renderEmpty) {
    return null
  }

  return (
    <ContribBoundary id={match.id}>
      <ContribRender render={renderEmpty} />
    </ContribBoundary>
  )
}

import type { FC } from 'react'
import { useMemo } from 'react'

import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { CHAT_EMPTY_AREA, type ChatEmptyContribution } from '@/lib/chat-empty'

/**
 * The empty transcript's contributed slot. Mounts every registration and lets
 * each decide — it renders the session's empty state, or nothing at all if the
 * session isn't one it owns.
 *
 * Deliberately a mount rather than a claim the transcript resolves up front:
 * whether a session has an empty state depends on data the plugin loads on its
 * own clock (a bot chat's roster lands after the transcript), and only a
 * mounted component can subscribe and appear when it arrives.
 *
 * That same asynchrony is why the slot cannot mount only the first
 * registration. Ownership is per session and is not known until each has
 * loaded, so first-wins would let a plugin that DECLINES a session suppress the
 * one that owns it, permanently and silently, purely on registration order.
 * Mounting all of them means disjoint owners each work; two claiming the same
 * session render both, which is a visible conflict rather than a silent drop.
 */
const ChatEmptyEntry: FC<{ id: string; render: ChatEmptyContribution['render']; sessionId: string }> = ({
  id,
  render,
  sessionId
}) => {
  // Stable component identity: ContribRender mounts this AS a component, so a
  // fresh closure per render would remount the empty state on every tick.
  const renderEmpty = useMemo(() => () => render({ sessionId }), [render, sessionId])

  return (
    <ContribBoundary id={id}>
      <ContribRender render={renderEmpty} />
    </ContribBoundary>
  )
}

export const ChatEmptySlot: FC<{ sessionId: string }> = ({ sessionId }) => {
  const contributions = useContributions(CHAT_EMPTY_AREA)

  return (
    <>
      {contributions.map(contribution => {
        const render = (contribution.data as ChatEmptyContribution | undefined)?.render

        return render ? (
          <ChatEmptyEntry id={contribution.id} key={contribution.id} render={render} sessionId={sessionId} />
        ) : null
      })}
    </>
  )
}

import type { FC } from 'react'
import { useMemo } from 'react'

import type { Contribution } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { type ThreadTailContribution, type ThreadTailProps } from '@/lib/thread-tail'

/**
 * The transcript tail's contributed slot. Mounts every registration and lets
 * each decide whether this session is one it has something to say in — the
 * same shape as `ChatEmptySlot`, for the same reason: ownership is per session
 * and not known until each plugin has loaded, so first-wins would let a
 * plugin that declines suppress the one that owns the tail.
 */
const ThreadTailEntry: FC<{ id: string; render: ThreadTailContribution['render']; sessionId: string }> = ({
  id,
  render,
  sessionId
}) => {
  // Stable component identity: ContribRender mounts this AS a component, so a
  // fresh closure per render would remount the tail on every tick.
  const renderTail = useMemo(() => () => render({ sessionId } satisfies ThreadTailProps), [render, sessionId])

  return (
    <ContribBoundary id={id} variant="chip">
      <ContribRender render={renderTail} />
    </ContribBoundary>
  )
}

export const ThreadTailSlot: FC<{ contributions: readonly Contribution[]; sessionId: string }> = ({
  contributions,
  sessionId
}) => (
  <>
    {contributions.map(contribution => {
      const render = (contribution.data as ThreadTailContribution | undefined)?.render

      return render ? (
        <ThreadTailEntry id={contribution.id} key={contribution.id} render={render} sessionId={sessionId} />
      ) : null
    })}
  </>
)

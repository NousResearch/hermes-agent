import { useStore } from '@nanostores/react'
import type { FC } from 'react'
import { useMemo } from 'react'

import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { type SessionRowSlotContribution, type SessionRowSlotProps } from '@/lib/session-row-slots'
import { $activeRoute, sessionRouteContext } from '@/store/session-route-context'
import type { SessionInfo } from '@/types/hermes'

/**
 * One row-decoration slot (leading / trailing) for the row's session. Mounts
 * every registration and lets each decide — it renders its decoration, or
 * nothing at all for rows it doesn't own.
 *
 * Mounting all of them (rather than first-wins) keeps ownership per session:
 * a plugin that declines a row must not suppress the one that owns it purely
 * on registration order. Two decorations on one row render both — a visible
 * composition, not a silent drop.
 */
const SessionRowSlotEntry: FC<{
  context: SessionRowSlotProps
  id: string
  render: SessionRowSlotContribution['render']
}> = ({ context, id, render }) => {
  // Stable component identity: ContribRender mounts this AS a component, so a
  // fresh closure per render would remount the decoration on every tick.
  const renderSlot = useMemo(() => () => render(context), [render, context])

  return (
    <ContribBoundary id={id} variant="chip">
      <ContribRender render={renderSlot} />
    </ContribBoundary>
  )
}

export const SessionRowSlot: FC<{ area: string; session: SessionInfo }> = ({ area, session }) => {
  const contributions = useContributions(area)
  const active = useStore($activeRoute)

  // Rebuilt only when the row's identity or the active route changes, so the
  // decoration's memoised render keeps its component identity across ticks.
  const lineageKey = `${session.id}\0${session._lineage_root_id ?? ''}\0${(session._lineage_ids ?? []).join('\0')}`

  const context = useMemo(
    () => sessionRouteContext(session, active),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lineageKey, session.profile, session.connection_id, active]
  )

  if (contributions.length === 0) {
    return null
  }

  return (
    <>
      {contributions.map(contribution => {
        const render = (contribution.data as SessionRowSlotContribution | undefined)?.render

        return render ? (
          <SessionRowSlotEntry
            context={context}
            id={contribution.id}
            key={`${contribution.source ?? 'core'}:${contribution.id}`}
            render={render}
          />
        ) : null
      })}
    </>
  )
}

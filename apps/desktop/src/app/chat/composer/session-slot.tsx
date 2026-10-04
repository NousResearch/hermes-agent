import { useStore } from '@nanostores/react'
import type { FC, ReactNode } from 'react'
import { useMemo } from 'react'

import { composerFloatingStrip } from '@/components/chat/composer-dock'
import { useContributions } from '@/contrib'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import type { SessionRouteContext } from '@/lib/session-row-slots'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $sessions } from '@/store/session'
import { $activeRoute, runtimeRouteContext } from '@/store/session-route-context'
import { $sessionStates } from '@/store/session-states'

import { COMPOSER_AREAS } from './contrib'

/** Payload of a `composer.session` data contribution — a strip above the
 *  status stack that describes THIS composer's conversation (linked work,
 *  provenance). `render` gets the conversation's route context, so the plugin
 *  never resolves lineage or ownership itself; return `null` to stay out. */
export interface ComposerSessionContribution {
  render: (context: SessionRouteContext) => ReactNode
}

const Entry: FC<{ context: SessionRouteContext; id: string; render: ComposerSessionContribution['render'] }> = ({
  context,
  id,
  render
}) => {
  const renderSlot = useMemo(() => () => render(context), [render, context])

  return (
    <ContribBoundary id={id} variant="chip">
      <ContribRender render={renderSlot} />
    </ContribBoundary>
  )
}

/** Renders every `composer.session` contribution for the composer's session.
 *  `sessionId` is the composer's RUNTIME id; a draft with no stored id yet has
 *  no conversation to describe, so nothing mounts. */
export const ComposerSessionSlot: FC<{ sessionId: null | string }> = ({ sessionId }) => {
  const contributions = useContributions(COMPOSER_AREAS.session)
  const sessions = useStore($sessions)
  const active = useStore($activeRoute)

  const stored = useStoreSelector($sessionStates, states =>
    sessionId ? (states[sessionId]?.storedSessionId ?? null) : null
  )

  const context = useMemo(
    () => runtimeRouteContext(stored, sessions, active, sessionId),
    [stored, sessions, active, sessionId]
  )

  if (contributions.length === 0 || !context) {
    return null
  }

  return (
    <div className={cn(composerFloatingStrip, 'px-[5px] pb-1.5 empty:hidden')} data-slot="composer-session-strip">
      {contributions.map(contribution => {
        const render = (contribution.data as ComposerSessionContribution | undefined)?.render

        return render ? (
          <Entry
            context={context}
            id={contribution.id}
            key={`${contribution.source ?? 'core'}:${contribution.id}`}
            render={render}
          />
        ) : null
      })}
    </div>
  )
}

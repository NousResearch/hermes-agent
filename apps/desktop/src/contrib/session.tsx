import { type ComponentType, createElement } from 'react'

import { ContribBoundary } from './react/boundary'
import { useContributions } from './react/use-contributions'

export const SESSION_AREAS = {
  statusStack: 'session.statusStack',
  tileBadge: 'session.tileBadge',
  listBadge: 'session.listBadge'
} as const

/** Credential-free identity of the session at the contribution's own surface. */
export interface PluginSessionContext {
  runtimeSessionId: string | null
  storedSessionId: string | null
  profile: string
  connectionId: string
}

export interface SessionContributionProps {
  session: PluginSessionContext
}

export interface SessionContribution {
  render: ComponentType<SessionContributionProps>
}

export function SessionContributionSlot({ area, session }: { area: string; session: PluginSessionContext | null }) {
  const contributions = useContributions(area)

  if (!session) {
    return null
  }

  return (
    <>
      {contributions.map(c => {
        const render = (c.data as SessionContribution | undefined)?.render

        return render ? (
          <ContribBoundary id={c.id} key={`${c.source}:${c.id}:${JSON.stringify(session)}`} variant="chip">
            {createElement(render, { session })}
          </ContribBoundary>
        ) : null
      })}
    </>
  )
}

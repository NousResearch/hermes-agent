import { createElement, useMemo } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import type { Contribution } from '@/contrib/types'
import {
  TOOL_ACTIVITY_HEADER_API_VERSION,
  TOOL_ACTIVITY_HEADER_AREA,
  type ToolActivityHeaderContext,
  type ToolActivityHeaderContribution
} from '@/lib/tool-activity-header'

interface HeaderProps {
  context: ToolActivityHeaderContext
}

function HeaderContent({ context, contribution }: HeaderProps & { contribution: Contribution }) {
  // Read untyped disk-plugin payloads INSIDE the boundary, including accessors.
  const data = contribution.data as ToolActivityHeaderContribution | undefined

  if (data?.apiVersion !== TOOL_ACTIVITY_HEADER_API_VERSION || typeof data.render !== 'function') {
    return null
  }

  // Give the callback its own React identity/hooks, with immutable props in production too.
  const element = createElement(data.render, context)
  Object.freeze(element.props)

  return element
}

function HeaderEntry({ context, contribution }: HeaderProps & { contribution: Contribution }) {
  // Reuse the host boundary. Its identity follows the registration, not the
  // changing context: state survives updates, and replacement retries failures.
  const Guarded = useMemo(
    () =>
      function GuardedHeader(props: HeaderProps) {
        return (
          <ErrorBoundary fallback={() => null} label={`contrib:${contribution.id}`}>
            <HeaderContent {...props} contribution={contribution} />
          </ErrorBoundary>
        )
      },
    [contribution]
  )

  return <Guarded context={context} />
}

/** Additive only: no wrapper DOM or native-row changes when no plugin contributes. */
export function ToolActivityHeader({ context }: HeaderProps) {
  const entries = useContributions(TOOL_ACTIVITY_HEADER_AREA)

  if (entries.length === 0) {
    return null
  }

  return (
    <>
      {entries.map(contribution => (
        <HeaderEntry context={context} contribution={contribution} key={contribution.id} />
      ))}
    </>
  )
}

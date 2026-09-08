import { type ComponentType, createElement, type ReactNode, useEffect, useState } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { useContributions } from '@/contrib'
import { createRendererLoopPauseController } from '@/lib/renderer-loop-pause'

export const SYSTEM_ACTIVITY_AREA = 'ui.systemActivity'

/** An operation owned by the app, never an Agent turn or a wait for a person. */
export interface SystemActivityProps {
  activity: 'loading' | 'connecting' | 'processing'
  placement: 'inline' | 'region' | 'threshold'
  /** Freeze the mark while its pane/window is hidden or its operation is leaving. */
  paused: boolean
  /** Host-owned status text, available to render visibly at an application threshold. */
  label?: string
  /** The host already renders progress for this same operation. */
  hasProgress?: boolean
}

export interface SystemActivityContribution {
  /** Decorative component; the host owns accessibility. Null deliberately omits the mark. */
  render: ComponentType<SystemActivityProps>
}

interface SystemActivitySlotProps extends Omit<SystemActivityProps, 'paused'> {
  fallback: ReactNode
  /** Omit when the surrounding status already supplies the accessible name. */
  label?: string
  paused?: boolean
}

/** Only the mark is replaceable; operation state, text and recovery stay with its caller. */
export function SystemActivitySlot({ fallback, label, ...props }: SystemActivitySlotProps) {
  const contributions = useContributions(SYSTEM_ACTIVITY_AREA)

  const owner = contributions.find(
    contribution => (contribution.data as SystemActivityContribution | undefined)?.render != null
  )

  const [boundary, setBoundary] = useState({ owner, revision: 0 })

  // A replacement registration must recover even if the previous renderer failed.
  if (boundary.owner !== owner) {
    setBoundary({ owner, revision: boundary.revision + 1 })
  }

  if (!owner) {
    return fallback
  }

  return (
    <ErrorBoundary fallback={() => fallback} key={boundary.revision} label={`contrib:${owner.id}`}>
      {label ? (
        <span aria-label={label} className="inline-flex" role="status">
          <ContributedActivity label={label} render={(owner.data as SystemActivityContribution).render} {...props} />
        </span>
      ) : (
        <ContributedActivity render={(owner.data as SystemActivityContribution).render} {...props} />
      )}
    </ErrorBoundary>
  )
}

function ContributedActivity({
  render,
  paused = false,
  ...props
}: Omit<SystemActivitySlotProps, 'fallback'> & SystemActivityContribution) {
  const paneVisible = usePaneVisible()
  const [windowPaused, setWindowPaused] = useState(() => document.hidden || !document.hasFocus())

  useEffect(() => {
    const controller = createRendererLoopPauseController(() => setWindowPaused(controller.isPaused()))
    setWindowPaused(controller.isPaused())

    return controller.dispose
  }, [])

  return createElement(render, { ...props, paused: paused || !paneVisible || windowPaused })
}

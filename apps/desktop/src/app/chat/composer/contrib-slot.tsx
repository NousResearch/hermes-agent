import type { ReactNode } from 'react'

import { ContribBoundary } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'

import type { ComposerRenderContext } from './contrib'

export interface ComposerRenderSlotProps {
  /** Composer render area id (`composer.actions`, `composer.top`, …). */
  area: string
  /** The composer's edit bridge, handed to every contribution's `render`. */
  ctx: ComposerRenderContext
}

/**
 * Renders a composer render-area contribution list (`composer.actions`,
 * `composer.top`, …) — the composer-specific sibling of the generic `Slot`,
 * differing in TWO ways: each contribution's `render` receives the composer's
 * edit bridge (`ComposerRenderContext`) as its first argument, so a
 * `composer.actions` plugin can insert text that lands on the app undo stack
 * through the app's own DOM pipeline; and the render call itself is deferred
 * into the boundary's subtree, so a plugin whose `render` throws directly
 * degrades to the inline slot error instead of escaping to the app root.
 * Contributions that ignore the context argument (a `render` written against
 * the generic `() => ReactNode` shape) keep working untouched.
 */
export function ComposerRenderSlot({ area, ctx }: ComposerRenderSlotProps) {
  const items = useContributions(area)

  if (items.length === 0) {
    return null
  }

  return (
    <>
      {items.map(c => (
        <ContribBoundary id={c.id} key={`${c.source ?? 'core'}:${c.id}`} variant="chip">
          <ContributionRender ctx={ctx} render={c.render} />
        </ContribBoundary>
      ))}
    </>
  )
}

interface ContributionRenderProps {
  render: (() => ReactNode) | undefined
  ctx: ComposerRenderContext
}

/** Invokes a contribution render with the composer edit bridge. The generic
 *  `Contribution.render` type can't know the area's context, so the bridge is
 *  narrowed here — the one place the two shapes meet. Being a component (not
 *  an inline call) also puts the invocation INSIDE the boundary subtree, so a
 *  direct throw from plugin render code is caught like any child crash. */
function ContributionRender({ render, ctx }: ContributionRenderProps) {
  return <>{render ? (render as (ctx?: ComposerRenderContext) => ReactNode)(ctx) : null}</>
}

/**
 * The presentation leaves every Bot Mode dialog shares: a labelled control, and
 * the frame the embedded Capabilities surface is rendered into.
 *
 * They sit below the dialogs rather than inside any one of them — the model
 * picker, the advanced editor, Edit Profile, New Bot and the routines dialogs
 * all render the same label-over-control pair, and none of them can own it
 * without the others importing a sibling surface.
 */

import type { ReactNode } from 'react'

export function labeled(label: ReactNode, control: ReactNode) {
  return (
    <div className="grid gap-1.5">
      <label className="text-xs font-medium text-(--ui-text-secondary)">{label}</label>
      {control}
    </div>
  )
}

interface ResizableFrameProps {
  children: ReactNode
  /** CSS `resize` has nothing to drag from without a concrete height, and the
   *  two surfaces that embed Capabilities budget it different room. */
  height: number
  minHeight: number
}

/** A fixed-height viewport with the native vertical resize handle. */
export function ResizableFrame({ children, height, minHeight }: ResizableFrameProps) {
  return (
    <div
      className="resize-y overflow-auto rounded-md border border-(--ui-stroke-secondary)"
      style={{
        height,
        minHeight
      }}
    >
      {children}
    </div>
  )
}

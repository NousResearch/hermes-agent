/**
 * The one presentation leaf every Bot Mode dialog shares: a labelled control.
 *
 * It sits below the dialogs rather than inside any one of them — the model
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

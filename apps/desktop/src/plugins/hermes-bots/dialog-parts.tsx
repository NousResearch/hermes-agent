/**
 * The presentation leaves every Bot Mode dialog shares: a labelled control, and
 * the frame the embedded Capabilities surface is rendered into.
 *
 * They sit below the dialogs rather than inside any one of them — the model
 * picker, the advanced editor, Edit Profile, New Bot and the routines dialogs
 * all render the same label-over-control pair, and none of them can own it
 * without the others importing a sibling surface.
 */

import { type ReactNode, useId } from 'react'

import { nameControl } from '@/lib/name-control'

/**
 * A field label above its control.
 *
 * The `<label>` here has no `htmlFor` and does not wrap the control, so it was
 * decoration only: every bot-dialog select read as a bare value with nothing
 * saying which field it was (SenseReader, 2026-09-05). `nameControl` joins the
 * two the same way the settings rows do — one helper, so the two layouts cannot
 * drift apart again.
 *
 * Still a function call, not a component, so the ~9 call sites stay as they
 * are; the component it returns is what owns the generated id.
 */
export function labeled(label: ReactNode, control: ReactNode) {
  return <Labeled control={control} label={label} />
}

function Labeled({ control, label }: { control: ReactNode; label: ReactNode }) {
  const labelId = useId()
  const { group, node } = nameControl(control, labelId)

  return (
    <div aria-labelledby={group ? labelId : undefined} className="grid gap-1.5" role={group ? 'group' : undefined}>
      <label className="text-xs font-medium text-(--ui-text-secondary)" id={labelId}>
        {label}
      </label>
      {node}
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

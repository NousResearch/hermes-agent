/**
 * The presentation leaves every Bot Mode dialog shares: a labelled control, and
 * the frame the embedded Capabilities surface is rendered into.
 *
 * They sit below the dialogs rather than inside any one of them — the model
 * picker, the advanced editor, Edit Profile, New Bot and the routines dialogs
 * all render the same label-over-control pair, and none of them can own it
 * without the others importing a sibling surface.
 */

import { Input, Textarea } from '@hermes/plugin-sdk'
import { cloneElement, isValidElement, type ReactNode, useId } from 'react'

export function labeled(label: ReactNode, control: ReactNode) {
  return <LabeledControl control={control} label={label} />
}

interface LabeledControlProps {
  control: ReactNode
  label: ReactNode
}

function LabeledControl({ control, label }: LabeledControlProps) {
  const generatedId = useId()

  // Composite pickers and capability groups have several controls; their
  // caption must not implicitly label whichever input happens to come first.
  const input =
    isValidElement<{ id?: string }>(control) &&
    (control.type === Input || control.type === Textarea || control.type === 'input' || control.type === 'textarea')
      ? control
      : null

  const inputId = input ? input.props.id || generatedId : undefined

  return (
    <div className="grid gap-1.5">
      <label className="text-xs font-medium text-(--ui-text-secondary)" htmlFor={inputId}>
        {label}
      </label>
      {input ? cloneElement(input, { id: inputId }) : control}
    </div>
  )
}

// Module-level so the ref callback is stable and fires on mount only.
const revealOnMount = (el: HTMLDivElement | null) => el?.scrollIntoView?.({ block: 'nearest' })

/** The embedded Capabilities viewport: tall enough for the installed list AND
 *  the Skills Hub picker stacked under it (a fixed few hundred pixels left the
 *  list a sliver), with the native vertical resize handle. Scrolled into view
 *  on mount, since it opens below the dialog's identity fields. */
export function ResizableFrame({ children }: { children: ReactNode }) {
  return (
    <div
      className="resize-y overflow-auto rounded-md border border-(--ui-stroke-secondary)"
      ref={revealOnMount}
      style={{
        height: 'min(85vh, 960px)',
        minHeight: 360
      }}
    >
      {children}
    </div>
  )
}

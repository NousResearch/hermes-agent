import { createContext, type ReactNode, useContext, useId } from 'react'

import { cn } from '@/lib/utils'

/** Label above the control. `Field` and `SidePanelMetaRow` share this so a
 *  dialog, the Kanban drawer, and the workflow inspector cannot drift. */
export const FIELD_STACK = 'grid gap-1.5'
export const FIELD_LABEL = 'flex gap-2 text-xs font-medium text-foreground'

/**
 * What a field has to say about its own value, said under the control rather
 * than in a pile at the top of the form.
 *
 * Two voices, because two things ask for one. A form you submit knows the
 * moment you're wrong and should say so: `error`. An editor with no submit —
 * the workflow inspector, where every keystroke is already saved — is looking
 * at a draft that is incomplete by definition, and a step you dropped a second
 * ago should not be shouting: `notice`, the same muted type as help text,
 * promoted to `error` when you ask to run and the draft has to be finished.
 */
export interface FieldStatus {
  level: 'notice' | 'error'
  message: ReactNode
}

interface FieldControl {
  describedBy: string
  invalid: boolean
}

const FieldControlContext = createContext<FieldControl | null>(null)

/** Props a control wears so an enclosing status reaches it — the invalid
 *  border (`.desktop-input-chrome[aria-invalid]`) and the message as its
 *  description. Spread BEFORE the caller's own props so those still win.
 *  `Input`, `Textarea` and `SelectTrigger` already do this; a control built on
 *  `controlVariants` should too. */
export function useFieldControl() {
  const ctx = useContext(FieldControlContext)

  if (!ctx) {
    return undefined
  }

  return { 'aria-describedby': ctx.describedBy, 'aria-invalid': ctx.invalid || undefined }
}

/**
 * A control plus whatever its field has to say about it. `Field` and
 * `SidePanelMetaRow` wrap their children in this, which is what keeps a dialog
 * and a side panel saying it the same way; use it directly for a control that
 * isn't in either (a bare textarea in a section, an unlabelled dialog input).
 */
export function FieldStatusSlot({
  children,
  hintClassName,
  status
}: {
  children: ReactNode
  /** Where the message sits when the surrounding grid isn't a plain stack. */
  hintClassName?: string
  status?: FieldStatus
}) {
  const id = useId()

  if (!status) {
    return <>{children}</>
  }

  return (
    <FieldControlContext.Provider value={{ describedBy: id, invalid: status.level === 'error' }}>
      {children}
      <FieldHint className={hintClassName} error={status.level === 'error'} id={id}>
        {status.message}
      </FieldHint>
    </FieldControlContext.Provider>
  )
}

// Shared form-field primitive: a label stacked above its control, with an
// optional inline "(optional)" tag. Pass `status` for what's wrong with the
// value, or pair with FieldHint for static help text below the control. This is
// the single field language for every form — dialogs (cron, webhooks,
// profiles), the Kanban drawer, the workflow inspector. Don't hand-roll
// label+control stacks, and don't hand-roll an error line under one. Stack
// Fields in a `grid gap-4` form; pair two across with
// `grid items-start gap-4 sm:grid-cols-2`.
export function Field({
  children,
  className,
  htmlFor,
  label,
  optional,
  optionalLabel,
  row,
  status,
  tip
}: {
  children: ReactNode
  className?: string
  htmlFor?: string
  label: ReactNode
  optional?: boolean
  optionalLabel?: string
  /** Label beside the control. For a small control (a stepper, a switch)
   *  that would look lost at full width. */
  row?: boolean
  /** What's wrong with the value — rendered under the control, and it marks
   *  the control invalid. */
  status?: FieldStatus
  /** Hover guidance. For a panel of knobs, where a `FieldHint` under every one
   *  would triple its height, this keeps the help off the surface until asked. */
  tip?: string
}) {
  // A <label> is only valid around ONE control; a segmented control or a
  // stepper is several, so those get a plain div and an unassociated caption.
  const Tag = htmlFor === undefined ? 'div' : 'label'

  return (
    <div
      className={cn(row ? 'grid grid-cols-[6rem_minmax(0,1fr)] items-center gap-x-3' : FIELD_STACK, className)}
      title={tip}
    >
      <Tag
        className={cn(FIELD_LABEL, row ? 'items-center' : 'items-baseline')}
        {...(htmlFor ? { htmlFor, id: `${htmlFor}-label` } : {})}
      >
        {label}
        {optional && optionalLabel && (
          <span className="text-[0.65rem] font-normal text-muted-foreground">{optionalLabel}</span>
        )}
      </Tag>
      {/* `row` puts the label in column one, so the message tracks the control
          in column two rather than starting back under the label. */}
      <FieldStatusSlot hintClassName={row ? 'col-start-2 mt-1' : undefined} status={status}>
        {children}
      </FieldStatusSlot>
    </div>
  )
}

export function FieldHint({
  children,
  className,
  error,
  id
}: {
  children: ReactNode
  className?: string
  error?: boolean
  id?: string
}) {
  return (
    <p
      className={cn('text-[0.66rem] leading-4', error ? 'text-destructive' : 'text-muted-foreground', className)}
      id={id}
    >
      {children}
    </p>
  )
}

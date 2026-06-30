import * as React from 'react'

/**
 * A full-row / full-region click target rendered as a real `<button>`.
 *
 * Several surfaces intentionally make an entire row (or cell) the click target
 * while hosting nested layout and controls inside it — sidebar rows, overlay /
 * panel list rows, settings + onboarding provider rows, the artifacts cell.
 * This primitive bakes in the shared semantics (`type="button"` plus a stable
 * `data-slot`) WITHOUT imposing any visual styling, so each row keeps its own
 * layout classes and nothing changes visually.
 *
 * Use `RowButton` for these row/region targets; reach for `Button`
 * (`components/ui/button.tsx`) for ordinary compact actions.
 */
function RowButton({ className, type = 'button', ...props }: React.ComponentProps<'button'>) {
  return <button className={className} data-slot="row-button" type={type} {...props} />
}

export { RowButton }

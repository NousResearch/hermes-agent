import { cloneElement, isValidElement, type ReactNode } from 'react'

/** Host elements that can take an accessible name directly. */
const LABELABLE_TAGS = new Set(['button', 'input', 'meter', 'output', 'progress', 'select', 'textarea'])

/**
 * Point a control at the text that labels it, when the two are siblings rather
 * than a `<label for>` pair.
 *
 * Settings rows and the bot dialogs both draw the label as its own node beside
 * the control, with nothing joining them — SenseReader read those controls as
 * bare values ("OpenRouter 콤보상자") with no way to tell which setting was
 * which (2026-09-05). Both now route through here, so the two layouts cannot
 * drift apart again.
 *
 * Two shapes, because naming the wrong node is the same as not naming it:
 *
 *  - a component, or a labelable host element → name it directly. A component
 *    forwards the prop to whatever control it renders.
 *  - a layout wrapper (a `<div>` holding a control, which is how several rows
 *    are built) → `aria-labelledby` on a `<div>` names nothing. The caller
 *    should make the CONTAINER a named group instead, which is what `group`
 *    reports, so a reader entering it still hears what these controls are for.
 *
 * A control that already names itself is left alone either way — its own label
 * was written for it, and the surrounding text is only a fallback.
 */
export function nameControl(control: ReactNode, labelId: string): { group: boolean; node: ReactNode } {
  if (!isValidElement(control)) {
    return { group: false, node: control }
  }

  const props = control.props as { 'aria-label'?: string; 'aria-labelledby'?: string }

  if (props['aria-label'] || props['aria-labelledby']) {
    return { group: false, node: control }
  }

  const type = control.type

  if (typeof type !== 'string' || LABELABLE_TAGS.has(type)) {
    return { group: false, node: cloneElement(control, { 'aria-labelledby': labelId } as Record<string, unknown>) }
  }

  return { group: true, node: control }
}

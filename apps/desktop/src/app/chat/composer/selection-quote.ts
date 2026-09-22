import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { setComposerTerminalSelection } from '@/store/composer'

import { requestComposerInsert } from './focus'

export const GUEST_SELECTION_CHANNEL = 'preview-selection-to-composer'

export function selectionQuote(text: string, label: string, editable: boolean) {
  const trimmed = text.trim()

  if (editable || !trimmed) {
    return null
  }

  return { label: label.trim() || 'Preview', text: trimmed }
}

/** Store a non-editable DOM/guest selection using the existing terminal-style
 * composer context contract. The neutral name is intentional: the composer
 * renders these as quoted context regardless of the pane that supplied them. */
export function sendSelectionQuote(text: string, label: string, editable = false): boolean {
  const quote = selectionQuote(text, label, editable)

  if (!quote) {
    return false
  }

  setComposerTerminalSelection(quote.label, quote.text)
  requestComposerInsert(`@terminal:${formatRefValue(quote.label)}`, { mode: 'inline' })

  return true
}

/** Reusable copy shapes referenced by the ``Translations`` tree.
 *
 * Extracted from ``types.ts`` (topical sibling; that file is over the
 * FILE_LINES cap and may only shrink).
 */

/** One error-card entry: a short title and one plain sentence. Either may
 *  take the failing provider's display name (falls back to "the AI service"). */
export interface ErrorCardCopy {
  title: string | ((provider: string) => string)
  body: string | ((provider: string) => string)
}

export interface ToolTitleCopy {
  done: string
  pending: string
  pendingAction: string
}

export interface ModeOptionCopy {
  label: string
  description: string
}

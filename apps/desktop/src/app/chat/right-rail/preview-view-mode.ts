// The preview rail's view-mode policy, extracted from LocalFilePreview so the
// Markdown-defaults-to-rendered contract is unit-testable without mounting the
// preview component (which pulls Streamdown, Shiki, CodeMirror and the desktop
// fs bridge).
//
// Policy: which toggles a text file offers, and which one it lands on when the
// user hasn't picked. Markdown renders formatted by default; an uncommitted
// diff always wins the default because reviewing changes beats reading; a
// user's explicit pick survives only while it is still offered.

export type PreviewViewMode = 'diff' | 'rendered' | 'source'

export interface ViewModeInputs {
  /** The file parses as Markdown — the rendered toggle exists. */
  isMarkdown: boolean
  /** A non-empty working-tree-vs-HEAD diff exists. */
  hasDiff: boolean
}

/**
 * Toggle order is also display order: rendered first (markdown only), then
 * source, then diff. The default lands on the most useful view.
 */
export function availableViewModes({ hasDiff, isMarkdown }: ViewModeInputs): PreviewViewMode[] {
  const modes: PreviewViewMode[] = []

  if (isMarkdown) {
    modes.push('rendered')
  }

  modes.push('source')

  if (hasDiff) {
    modes.push('diff')
  }

  return modes
}

/**
 * Which mode the preview shows: the user's explicit pick while it is still
 * available, else the automatic default (diff beats rendered beats source).
 */
export function resolveViewMode(inputs: ViewModeInputs, userPick: null | PreviewViewMode): PreviewViewMode {
  const modes = availableViewModes(inputs)

  if (userPick && modes.includes(userPick)) {
    return userPick
  }

  return inputs.hasDiff ? 'diff' : inputs.isMarkdown ? 'rendered' : 'source'
}

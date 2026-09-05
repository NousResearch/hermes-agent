// Pure decision helpers for file-tree row gestures.
//
// The file browser's interaction contract (user-confirmed): a single click
// opens a file in the in-app Preview pane; folders toggle instead. Extracted
// out of the React component so the gesture contract is unit-testable
// without mounting react-arborist, and so the rename/placeholder guards live
// in exactly one place.

/** Stable state a row hands us to decide what a gesture means. */
export interface FileRowState {
  isFolder: boolean
  isPlaceholder: boolean
  /** True while an inline rename edit is active on this row. */
  isRenaming: boolean
  /** True when Shift was held during the click (attach gestures). */
  shiftKey?: boolean
}

export type FileRowClickAction = 'attach-file' | 'attach-folder' | 'ignore' | 'open' | 'toggle'

/**
 * What a single click on the row should do. A file opens in preview;
 * a folder toggles its expansion. Shift-click attaches (drags into the
 * composer) rather than opening. Rename-in-progress and placeholders swallow
 * the click so the editor isn't yanked away mid-edit.
 *
 * Note: there is deliberately no double-click policy here. The row's click
 * handler stops propagation (suppressing arborist's own click handling), so
 * a double-click arrives as two ordinary clicks — and opening the same file
 * twice is idempotent (openPreview re-fronts the existing tab).
 */
export function resolveFileRowClick(state: FileRowState): FileRowClickAction {
  // A click on a placeholder row (e.g. the "no files" affordance) or while
  // an inline rename input is mounted must never select/open the underlying
  // file — the user is interacting with the placeholder or the editor.
  if (state.isPlaceholder || state.isRenaming) {
    return 'ignore'
  }

  if (state.shiftKey) {
    return state.isFolder ? 'attach-folder' : 'attach-file'
  }

  return state.isFolder ? 'toggle' : 'open'
}

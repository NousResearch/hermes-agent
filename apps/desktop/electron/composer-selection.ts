interface ComposerSelectionContext {
  canCompose: boolean
  isEditable: boolean
  selectionText: string
}

export interface ComposerSelectionMenuItem {
  click: () => void
  label: string
}

/**
 * Build the native menu action that forwards selected, read-only text to the
 * renderer composer. Editable fields keep Electron's normal cut/copy/paste
 * menu and never get this action.
 */
export function createComposerSelectionMenuItem(
  context: ComposerSelectionContext,
  sendSelection: (text: string) => void,
  reportError: (message: string) => void
): ComposerSelectionMenuItem | null {
  if (!context.canCompose || context.isEditable || !context.selectionText.trim()) {
    return null
  }

  return {
    label: 'Send Selection to Composer',
    click: () => {
      try {
        sendSelection(context.selectionText)
      } catch (error) {
        reportError(error instanceof Error ? error.message : String(error))
      }
    }
  }
}

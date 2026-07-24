interface SaveQueuedEditFromEditorArgs {
  editor: HTMLDivElement | null
  exitQueuedEdit: (action: 'save') => boolean
  flushEditorToDraft: (editor: HTMLDivElement) => void
}

export function saveQueuedEditFromEditor({
  editor,
  exitQueuedEdit,
  flushEditorToDraft
}: SaveQueuedEditFromEditorArgs): boolean {
  if (editor) {
    flushEditorToDraft(editor)
  }

  return exitQueuedEdit('save')
}

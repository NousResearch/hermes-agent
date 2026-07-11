interface PreviewWorkspaceContext {
  chatOpen: boolean
  hasPreviewSurfaces: boolean
  routedSessionId: string | null
  selectedStoredSessionId: string | null
}

export function shouldShowPreviewWorkspace({
  chatOpen,
  hasPreviewSurfaces,
  routedSessionId,
  selectedStoredSessionId
}: PreviewWorkspaceContext): boolean {
  return chatOpen && hasPreviewSurfaces && Boolean(selectedStoredSessionId || routedSessionId)
}

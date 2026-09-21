export async function readFilePreviewDataUrl(
  read: () => Promise<string>
): Promise<string> {
  try {
    return await read()
  } catch (error) {
    const code =
      error && typeof error === 'object' && 'code' in error
        ? String(error.code)
        : ''

    // Historical chats and the artifacts view can retain references to local
    // media that has since been moved or deleted. A missing preview is an
    // expected UI state, not an IPC failure. Keep all other filesystem and
    // hardening errors strict.
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      return ''
    }

    throw error
  }
}
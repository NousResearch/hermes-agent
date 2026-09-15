type OpenFolder = (path: string) => Promise<void> | void

/** Resolve the one native directory shape that the desktop shell owns.
 * Everything else remains unclaimed so composer/file drop targets can handle it. */
export function nativeFolderPathFromDrop(transfer: DataTransfer | null): string | null {
  if (!transfer?.items || transfer.items.length !== 1) {
    return null
  }

  const item = transfer.items[0]

  if (!item || item.kind !== 'file') {
    return null
  }

  try {
    const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null

    if (!entry?.isDirectory) {
      return null
    }

    const file = item.getAsFile()
    const getPath = window.hermesDesktop?.getPathForFile

    if (!file || !getPath) {
      return null
    }

    const path = getPath(file)?.trim()

    return path || null
  } catch {
    return null
  }
}

/** Install the app-wide capture listener that promotes native folder drops to
 * workspaces before nested file/composer targets can claim them. */
export function installDesktopFolderDrop(openFolder: OpenFolder): () => void {
  const onDrop = (event: DragEvent) => {
    const path = nativeFolderPathFromDrop(event.dataTransfer)

    if (!path) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    void openFolder(path)
  }

  window.addEventListener('drop', onDrop, true)

  return () => window.removeEventListener('drop', onDrop, true)
}

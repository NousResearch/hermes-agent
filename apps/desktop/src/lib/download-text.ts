/** Save generated text content to a file via a blob download. Electron's
 *  default will-download behavior shows the OS save dialog, so this works
 *  without a dedicated IPC handler (same pattern as use-image-download's
 *  browser fallback). */
export function downloadTextFile(name: string, content: string, mimeType = 'text/plain') {
  const blobUrl = URL.createObjectURL(new Blob([content], { type: `${mimeType};charset=utf-8` }))
  const link = document.createElement('a')

  link.href = blobUrl
  link.download = name
  link.rel = 'noopener noreferrer'
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 30_000)
}

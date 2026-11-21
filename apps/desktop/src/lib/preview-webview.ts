/** Construct an unprivileged guest, never the app renderer. Isolated viewers
 * get a fresh memory-only session; URLs/tickets never name the partition.
 * Other tabs ride `browserPartition`, the tab's per-profile Browser cookie jar. */
export function createPreviewWebview(
  url: string,
  browserContext?: 'isolated',
  browserPartition = 'persist:hermes-preview'
): HTMLElement {
  const webview = document.createElement('webview')
  // A page with no background of its own shows white, as in a browser and the remote HTML frame.
  webview.className = 'flex h-full w-full flex-1 bg-white'
  webview.setAttribute(
    'partition',
    browserContext === 'isolated' ? `hermes-viewer-${crypto.randomUUID()}` : browserPartition
  )
  webview.setAttribute('src', url)
  webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

  return webview
}

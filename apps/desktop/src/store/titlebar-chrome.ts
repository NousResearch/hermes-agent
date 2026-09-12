const TITLEBAR_CHROME_REVISION_EVENT = 'hermes:titlebar-chrome-revision'

let revision = 0

export function emitTitlebarChromeRevision(): void {
  revision += 1
  window.dispatchEvent(new CustomEvent(TITLEBAR_CHROME_REVISION_EVENT, { detail: revision }))
}

export function subscribeTitlebarChromeRevision(listener: () => void): () => void {
  const onRevision = () => listener()
  window.addEventListener(TITLEBAR_CHROME_REVISION_EVENT, onRevision)

  return () => window.removeEventListener(TITLEBAR_CHROME_REVISION_EVENT, onRevision)
}

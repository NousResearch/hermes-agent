const PREVIEW_ALLOWED_PROTOCOLS = new Set(['http:', 'https:', 'file:', 'hermes-media:'])
const PREVIEW_EXTERNAL_OPEN_PROTOCOLS = new Set(['http:', 'https:'])

function parsePreviewUrl(rawUrl) {
  try {
    return new URL(String(rawUrl || '').trim())
  } catch {
    return null
  }
}

function isAllowedPreviewNavigation(rawUrl) {
  const parsed = parsePreviewUrl(rawUrl)

  return Boolean(parsed && PREVIEW_ALLOWED_PROTOCOLS.has(parsed.protocol))
}

function isAllowedPreviewExternalOpen(rawUrl) {
  const parsed = parsePreviewUrl(rawUrl)

  return Boolean(parsed && PREVIEW_EXTERNAL_OPEN_PROTOCOLS.has(parsed.protocol))
}

function guardPreviewWebContents(webContents, { openExternalUrl } = {}) {
  if (!webContents) {
    return
  }

  const blockUnsafeNavigation = (event, url) => {
    if (isAllowedPreviewNavigation(url)) {
      return
    }

    event.preventDefault()
  }

  webContents.setWindowOpenHandler?.(details => {
    if (isAllowedPreviewExternalOpen(details.url) && typeof openExternalUrl === 'function') {
      openExternalUrl(details.url)
    }

    return { action: 'deny' }
  })

  webContents.on?.('will-navigate', blockUnsafeNavigation)
  webContents.on?.('will-frame-navigate', blockUnsafeNavigation)
  webContents.on?.('will-redirect', blockUnsafeNavigation)
}

module.exports = {
  PREVIEW_ALLOWED_PROTOCOLS,
  guardPreviewWebContents,
  isAllowedPreviewExternalOpen,
  isAllowedPreviewNavigation
}

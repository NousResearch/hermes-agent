/** Return an Origin only for renderer URLs owned by this Desktop app. */
export function trustedRendererOrigin(rendererUrl: string): string | undefined {
  let url: URL

  try {
    url = new URL(rendererUrl)
  } catch {
    return undefined
  }

  if (url.protocol === 'file:') {
    return undefined
  }

  if (url.protocol === 'hermes:' && url.host) {
    return `${url.protocol}//${url.host}`
  }

  if (url.protocol === 'http:' && url.hostname === '127.0.0.1' && url.port) {
    return url.origin
  }

  return undefined
}

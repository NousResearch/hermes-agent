export const FOLDER_LINK_PREFIX = '#folder/'

export function folderMarkdownHref(path: string): string {
  return (
    FOLDER_LINK_PREFIX +
    encodeURIComponent(path).replace(/[()]/g, c => `%${c.charCodeAt(0).toString(16).toUpperCase()}`)
  )
}

export function folderPathFromMarkdownHref(href?: string): string | null {
  if (!href?.startsWith(FOLDER_LINK_PREFIX)) {
    return null
  }

  const payload = href.slice(FOLDER_LINK_PREFIX.length)

  if (!/^(?:[\w.!~*'-]|%[\da-f]{2})+$/i.test(payload)) {
    return null
  }

  try {
    const path = decodeURIComponent(payload)

    // eslint-disable-next-line no-control-regex -- Native paths must not contain control characters.
    if (/[\x00-\x1f\x7f]/.test(path) || /^[\\/]{2}/.test(path)) {
      return null
    }

    return /^(?:\/(?!\/)|[a-z]:[\\/])/i.test(path) ? path : null
  } catch {
    return null
  }
}

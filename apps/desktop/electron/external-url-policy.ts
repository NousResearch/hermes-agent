const SAFE_RESPONSE_FILE_EXTENSIONS = new Set([
  '.bmp',
  '.csv',
  '.docx',
  '.flac',
  '.gif',
  '.heic',
  '.jpeg',
  '.jpg',
  '.json',
  '.log',
  '.m4a',
  '.m4v',
  '.markdown',
  '.md',
  '.mov',
  '.mp3',
  '.mp4',
  '.odp',
  '.ods',
  '.odt',
  '.ogg',
  '.pdf',
  '.png',
  '.pptx',
  '.rtf',
  '.tif',
  '.tiff',
  '.toml',
  '.txt',
  '.wav',
  '.webm',
  '.webp',
  '.xlsx',
  '.yaml',
  '.yml',
  '.zip'
])

function hasSafeResponseFileExtension(url: URL): boolean {
  try {
    const pathname = decodeURIComponent(url.pathname).toLowerCase()
    const extensionIndex = pathname.lastIndexOf('.')

    return extensionIndex >= 0 && SAFE_RESPONSE_FILE_EXTENSIONS.has(pathname.slice(extensionIndex))
  } catch {
    return false
  }
}

export type ExternalUrlClassification = {
  kind: 'external' | 'file'
  url: URL
}

export type WslExternalOpenCommand = {
  executable: 'rundll32.exe'
  args: ['url.dll,FileProtocolHandler', string]
}

export function wslExternalOpenCommand(url: string): WslExternalOpenCommand {
  return {
    executable: 'rundll32.exe',
    args: ['url.dll,FileProtocolHandler', url]
  }
}

function parseExternalUrl(rawUrl: unknown): URL | null {
  const raw = String(rawUrl || '').trim()

  if (!raw) {
    return null
  }

  try {
    return new URL(raw)
  } catch {
    return null
  }
}

function isValidatedLocalFileUrl(url: URL): boolean {
  return (
    url.protocol === 'file:' &&
    (!url.hostname || url.hostname.toLowerCase() === 'localhost') &&
    !url.username &&
    !url.password &&
    !url.port &&
    !url.pathname.startsWith('//') &&
    !/%2f|%5c/i.test(url.pathname)
  )
}

export function classifyExternalUrl(rawUrl: unknown): ExternalUrlClassification | null {
  const url = parseExternalUrl(rawUrl)

  if (!url) {
    return null
  }

  if (isValidatedLocalFileUrl(url)) {
    return { kind: 'file', url }
  }

  if (['http:', 'https:', 'mailto:'].includes(url.protocol)) {
    return { kind: 'external', url }
  }

  return null
}

export function classifyResponseLinkUrl(rawUrl: unknown): ExternalUrlClassification | null {
  const url = parseExternalUrl(rawUrl)

  if (!url) {
    return null
  }

  if (isValidatedLocalFileUrl(url) && hasSafeResponseFileExtension(url)) {
    return { kind: 'file', url }
  }

  if (
    url.protocol === 'obsidian:' &&
    url.hostname.toLowerCase() === 'open' &&
    !url.username &&
    !url.password &&
    !url.port &&
    (!url.pathname || url.pathname === '/') &&
    !url.hash
  ) {
    return { kind: 'external', url }
  }

  return null
}

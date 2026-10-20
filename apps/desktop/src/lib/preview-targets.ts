import { isWindowsAbsolutePath } from '@/lib/path-compare'

const PREVIEW_MARKDOWN_RE = /\[Preview:[^\]]+\]\((?<href>#preview[:/][^)]+)\)/gi

export function stripPreviewTargets(text: string): string {
  return text.replace(PREVIEW_MARKDOWN_RE, '')
}

export function extractPreviewTargets(text: string): string[] {
  const targets: string[] = []
  const seen = new Set<string>()

  for (const match of text.matchAll(PREVIEW_MARKDOWN_RE)) {
    const target = previewTargetFromMarkdownHref(match.groups?.href)

    if (target && !seen.has(target)) {
      seen.add(target)
      targets.push(target)
    }
  }

  return targets
}

export function previewMarkdownHref(target: string): string {
  return `#preview/${encodeURIComponent(target)}`
}

export function previewTargetFromMarkdownHref(href?: string): string | null {
  if (!href?.startsWith('#preview:') && !href?.startsWith('#preview/')) {
    return null
  }

  try {
    return decodeURIComponent(href.slice('#preview'.length + 1))
  } catch {
    return null
  }
}

// A filesystem path the agent linked mid-sentence (`[report](/tmp/report.md)`).
// Distinct from `#preview/` (a delivery card): this door renders as inline
// link text so the sentence around it still reads as a sentence. Same
// hardening-safe hash-href trick; the path resolves at click time.
const FILE_LINK_HREF_PREFIX = '#file/'

export function fileLinkMarkdownHref(path: string): string {
  return `${FILE_LINK_HREF_PREFIX}${encodeURIComponent(path)}`
}

export function fileLinkPathFromMarkdownHref(href?: string): string | null {
  if (!href?.startsWith(FILE_LINK_HREF_PREFIX)) {
    return null
  }

  try {
    return decodeURIComponent(href.slice(FILE_LINK_HREF_PREFIX.length)) || null
  } catch {
    return null
  }
}

export function previewName(target: string): string {
  // `new URL('C:\\...')` would read the drive letter as a URL scheme.
  if (isWindowsAbsolutePath(target)) {
    return target.split(/[\\/]/).filter(Boolean).pop() || target
  }

  try {
    const url = new URL(target)

    if (url.protocol === 'file:') {
      return decodeURIComponent(url.pathname).split(/[\\/]/).filter(Boolean).pop() || target
    }

    const file = url.pathname.split('/').filter(Boolean).pop()

    return file || url.host
  } catch {
    return target.split(/[\\/]/).filter(Boolean).pop() || target
  }
}

/** File identity only: do not resolve symlinks, guess home, or fold path case. */
export function previewArtifactKey(target: string, cwd: string): string {
  let path = target.trim()

  if (/^https?:\/\//i.test(path)) {
    return path
  }

  if (/^file:\/\//i.test(path)) {
    try {
      const url = new URL(path)

      // Encoded separators are not legal file-URL path segments.
      if (/%2f|%5c/i.test(url.pathname)) {
        return path
      }

      path = decodeURIComponent(url.pathname)

      if (url.hostname) {
        path = `//${url.hostname}${path}`
      } else if (/^\/[a-z]:\//i.test(path)) {
        path = path.slice(1)
      }
    } catch {
      return path
    }
  }

  const windows = /^[a-z]:[\\/]/i.test(path) || path.startsWith('\\\\')

  if (windows) {
    path = path.replace(/\\/g, '/')
  }

  if (!/^(?:\/|~\/|[a-z]:\/)/i.test(path) && cwd) {
    path = `${cwd.replace(/\/$/, '')}/${path.replace(/^\.\//, '')}`
  }

  if (/^[a-z]:[\\/]/i.test(path) || path.startsWith('\\\\')) {
    path = path.replace(/\\/g, '/')
  }

  return path.replace(/\/\.\//g, '/')
}

export function previewDisplayLabel(target: string): string {
  const escaped = previewName(target).replace(/[[\]\\]/g, '\\$&')

  return `Preview: ${escaped}`
}

const PREVIEW_MARKDOWN_RE = /\[Preview:[^\]]+\]\((?<href>#preview[:/][^)]+)\)/gi

export function stripPreviewTargets(text: string): string {
  return text
    .replace(PREVIEW_MARKDOWN_RE, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
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

function isNetworkPath(value: string) {
  return value.replaceAll('\\', '/').startsWith('//')
}

export function markdownArtifactTargetFromHref(href?: string): string | null {
  const target = href?.trim()

  if (!target) {
    return null
  }

  const windowsPath = /^[a-z]:[\\/]/i.test(target)
  const scheme = /^[a-z][a-z0-9+.-]*:/i.exec(target)?.[0].toLowerCase()
  let unsafeFileUrl = false

  if (scheme === 'file:') {
    try {
      const url = new URL(target)
      const hostname = url.hostname.toLowerCase()
      const pathname = decodeURIComponent(url.pathname)

      unsafeFileUrl = Boolean((hostname && hostname !== 'localhost') || isNetworkPath(pathname))
    } catch {
      return null
    }
  }

  if (
    target.startsWith('#') ||
    isNetworkPath(target) ||
    unsafeFileUrl ||
    (!windowsPath && scheme && scheme !== 'file:')
  ) {
    return null
  }

  const path = target.split(/[?#]/, 1)[0] || ''

  return /\.(?:md|markdown)$/i.test(path) ? target : null
}

export function markdownArtifactFileTarget(href?: string): string | null {
  const target = markdownArtifactTargetFromHref(href)

  if (!target) {
    return null
  }

  if (/^file:/i.test(target)) {
    try {
      const url = new URL(target)
      url.hash = ''
      url.search = ''

      return url.toString()
    } catch {
      return null
    }
  }

  return target.split(/[?#]/, 1)[0] || null
}

export function markdownArtifactHref(target: string): string {
  return `#artifact:${encodeURIComponent(target)}`
}

export function markdownArtifactTargetFromMarker(href?: string): string | null {
  if (!href?.startsWith('#artifact:')) {
    return null
  }

  try {
    return decodeURIComponent(href.slice('#artifact:'.length))
  } catch {
    return null
  }
}

export interface MarkdownArtifactAstNode {
  children?: unknown
  type?: unknown
  url?: unknown
}

export function remarkMarkdownArtifactLinks() {
  return (tree: unknown): void => {
    const visit = (node: unknown): void => {
      if (!node || typeof node !== 'object') {
        return
      }

      const candidate = node as MarkdownArtifactAstNode

      if ((candidate.type === 'link' || candidate.type === 'definition') && typeof candidate.url === 'string') {
        const target = markdownArtifactTargetFromHref(candidate.url)

        if (target) {
          candidate.url = markdownArtifactHref(target)
        }
      }

      if (Array.isArray(candidate.children)) {
        for (const child of candidate.children) {
          visit(child)
        }
      }
    }

    visit(tree)
  }
}

export function previewName(target: string): string {
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

export function previewDisplayLabel(target: string): string {
  const escaped = previewName(target).replace(/[[\]\\]/g, '\\$&')

  return `Preview: ${escaped}`
}

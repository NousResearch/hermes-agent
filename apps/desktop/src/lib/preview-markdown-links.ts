import { isWindowsAbsolutePath } from '@/lib/path-compare'

/**
 * Where a click in the desktop markdown preview should go.
 *
 * The preview is app UI, but it must not navigate on its own. Electron denies
 * every `target=_blank` / `window.open` (GHSA-9f4c-93c8-jc8g), so an https
 * anchor that looks clickable does nothing. A bare `#fragment` href is the
 * app's HashRouter, so a table-of-contents click changes the route instead of
 * scrolling the note. Callers turn this into `openLink`, an in-preview scroll,
 * or a sibling-file preview — never into `window.open`.
 */
export type PreviewMarkdownLink =
  | { kind: 'external'; href: string }
  | { kind: 'hash'; fragment: string }
  | { kind: 'file'; path: string }
  | { kind: 'inert' }

const BLOCKED_SCHEME = /^(?:javascript|data|blob|vbscript|chrome|chrome-extension|about|hermes):/i
const EXPLICIT_EXTERNAL = /^(?:https?:\/\/|mailto:)/i
const OTHER_SCHEME = /^[a-z][a-z0-9+.-]*:/i
const HEADING_SELECTOR = 'h1, h2, h3, h4, h5, h6'

export function decodeHashFragment(href: string): string {
  const raw = href.replace(/^#/, '')

  try {
    return decodeURIComponent(raw)
  } catch {
    return raw
  }
}

/** GitHub-style heading slug, which is what generated TOCs address. */
export function githubHeadingSlug(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\-_ ]+/gu, '')
    .replace(/ +/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

export function nextHeadingId(text: string, counts: Map<string, number>): string | null {
  const base = githubHeadingSlug(text)

  if (!base) {
    return null
  }

  const seen = counts.get(base) ?? 0

  counts.set(base, seen + 1)

  return seen === 0 ? base : `${base}-${seen}`
}

function normalizeSlashes(value: string): string {
  return value.replace(/\\/g, '/')
}

export function directoryOfFile(filePath: string): string {
  const normalized = normalizeSlashes(filePath)
  const slash = normalized.lastIndexOf('/')

  if (slash < 0) {
    return ''
  }

  if (slash === 0) {
    return '/'
  }

  const dir = normalized.slice(0, slash)

  if (/^[A-Za-z]:$/.test(dir)) {
    return `${dir}/`
  }

  return dir
}

export function joinPreviewPath(directory: string, relative: string): string {
  const base = normalizeSlashes(directory)
  const unc = base.startsWith('//')
  const windows = /^[A-Za-z]:(?:\/|$)/.test(base)
  const absolute = base.startsWith('/')
  const prefix = unc ? '//' : windows ? base.slice(0, 2) : absolute ? '/' : ''
  const rest = unc || windows ? base.slice(2) : absolute ? base.slice(1) : base
  const parts = rest.split('/').filter(Boolean)

  for (const part of normalizeSlashes(relative).split('/')) {
    if (!part || part === '.') {
      continue
    }

    if (part === '..') {
      parts.pop()

      continue
    }

    parts.push(part)
  }

  if (unc) {
    return `//${parts.join('/')}`
  }

  if (windows) {
    return `${prefix}/${parts.join('/')}`
  }

  if (absolute) {
    return `/${parts.join('/')}`
  }

  return parts.join('/')
}

function fileUrlToPath(href: string): string | null {
  try {
    const url = new URL(href)

    if (url.protocol !== 'file:') {
      return null
    }

    let path = decodeURIComponent(url.pathname)

    if (/^\/[A-Za-z]:\//.test(path)) {
      path = path.slice(1)
    }

    return path || null
  } catch {
    return null
  }
}

/** Path a preview link names, resolved against the note that contains it. */
export function resolvePreviewFileHref(href: string, filePath?: string): string | null {
  const stripped = href.trim().split(/[?#]/, 1)[0] ?? ''

  if (!stripped) {
    return null
  }

  if (/^file:/i.test(stripped)) {
    return fileUrlToPath(stripped)
  }

  let decoded = stripped

  try {
    decoded = decodeURIComponent(stripped)
  } catch {
    decoded = stripped
  }

  if (decoded === '~' || decoded.startsWith('~/') || decoded.startsWith('~\\')) {
    return decoded
  }

  if (decoded.startsWith('/') || isWindowsAbsolutePath(decoded)) {
    return normalizeSlashes(decoded)
  }

  if (!filePath) {
    return null
  }

  const directory = directoryOfFile(filePath)

  if (!directory) {
    return null
  }

  return joinPreviewPath(directory, decoded)
}

const PREVIEW_FILE_HOST = 'preview-file.invalid'

/** https sentinel for a resolved note path.
 *
 *  Streamdown's link hardener blocks `file:` and resolves `../note.md` against
 *  a dummy origin, keeping only `/note.md`. An https URL on a host nothing
 *  resolves survives that pass; classify turns it back into a file preview.
 */
export function previewFileSentinel(path: string): string {
  const normalized = path.replace(/\\/g, '/')

  const encoded = normalized
    .split('/')
    .map(part => encodeURIComponent(part))
    .join('/')

  return `https://${PREVIEW_FILE_HOST}${encoded.startsWith('/') ? encoded : `/${encoded}`}`
}

export function pathFromPreviewFileSentinel(href: string): string | null {
  try {
    const url = new URL(href)

    if (url.protocol !== 'https:' || url.hostname !== PREVIEW_FILE_HOST) {
      return null
    }

    let path = decodeURIComponent(url.pathname)

    if (/^\/[A-Za-z]:\//.test(path)) {
      path = path.slice(1)
    }

    return path || null
  } catch {
    return null
  }
}

interface MdastNode {
  children?: MdastNode[]
  type: string
  url?: string
}

export function rewritePreviewFileLinks(tree: MdastNode, filePath?: string): void {
  const visit = (node: MdastNode | undefined) => {
    if (!node || typeof node !== 'object') {
      return
    }

    if (node.type === 'link' && node.url && !pathFromPreviewFileSentinel(node.url)) {
      const decision = classifyPreviewMarkdownHref(node.url, filePath)

      if (decision.kind === 'file') {
        node.url = previewFileSentinel(decision.path)
      }
    }

    for (const child of node.children ?? []) {
      visit(child)
    }
  }

  visit(tree)
}

/** Remark attacher. Pass as `[remarkPreviewFileLinks, { filePath }]` so
 *  Streamdown's processor cache keys on the path. A closure over `filePath`
 *  keeps one function name and reuses the first note's processor for every
 *  later file. */
export function remarkPreviewFileLinks(options?: { filePath?: string }) {
  const filePath = options?.filePath

  return (tree: MdastNode) => {
    rewritePreviewFileLinks(tree, filePath)
  }
}

export function classifyPreviewMarkdownHref(href: string | undefined, filePath?: string): PreviewMarkdownLink {
  const raw = (href ?? '').trim()
  const sentinelPath = pathFromPreviewFileSentinel(raw)

  if (sentinelPath) {
    return { kind: 'file', path: sentinelPath }
  }

  if (!raw || BLOCKED_SCHEME.test(raw)) {
    return { kind: 'inert' }
  }

  if (raw.startsWith('#')) {
    const fragment = decodeHashFragment(raw)

    return fragment ? { kind: 'hash', fragment } : { kind: 'inert' }
  }

  if (raw.startsWith('//')) {
    return { kind: 'external', href: `https:${raw}` }
  }

  if (EXPLICIT_EXTERNAL.test(raw) || /^www\./i.test(raw)) {
    return { kind: 'external', href: /^www\./i.test(raw) ? `https://${raw}` : raw }
  }

  if (OTHER_SCHEME.test(raw) && !/^file:/i.test(raw)) {
    return { kind: 'external', href: raw }
  }

  const path = resolvePreviewFileHref(raw, filePath)

  return path ? { kind: 'file', path } : { kind: 'inert' }
}

/** Ids are stamped after render so they follow the heading text Streamdown
 *  actually painted, including duplicates (`slug`, `slug-1`). */
export function stampPreviewHeadingIds(root: ParentNode): void {
  const counts = new Map<string, number>()

  for (const heading of root.querySelectorAll<HTMLElement>(HEADING_SELECTOR)) {
    const id = nextHeadingId(heading.textContent ?? '', counts)

    if (id) {
      heading.id = id
    } else {
      heading.removeAttribute('id')
    }
  }
}

export function findPreviewHeading(root: ParentNode, fragment: string): HTMLElement | null {
  const decoded = decodeHashFragment(fragment)

  if (!decoded) {
    return null
  }

  const headings = [...root.querySelectorAll<HTMLElement>(HEADING_SELECTOR)]
  const slugged = githubHeadingSlug(decoded)

  return (
    headings.find(heading => heading.id === decoded) ??
    headings.find(heading => heading.id === slugged || heading.id.toLowerCase() === decoded.toLowerCase()) ??
    null
  )
}

export function scrollPreviewHeading(root: HTMLElement, fragment: string): boolean {
  const heading = findPreviewHeading(root, fragment)

  if (!heading) {
    return false
  }

  heading.scrollIntoView({ block: 'start' })

  return true
}

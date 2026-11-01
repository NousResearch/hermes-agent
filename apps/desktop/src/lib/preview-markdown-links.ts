import { joinPath } from '@/lib/local-preview'
import { previewMarkdownHref } from '@/lib/preview-targets'

/**
 * Where a click in the desktop markdown preview goes.
 *
 * Electron denies every `target=_blank` / `window.open` (GHSA-9f4c-93c8-jc8g),
 * so a web link has to travel through `ExternalLink` like chat links do. A
 * `#fragment` href is the app's HashRouter, so a table-of-contents click must
 * scroll the note instead of changing the route. A path to a sibling note is
 * protected from Streamdown's URL hardening by the same `#preview/…` door
 * chat transcripts use, and resolved against the note's directory on click.
 */
export type PreviewMarkdownLink =
  | { kind: 'external'; href: string }
  | { kind: 'file'; target: string }
  | { kind: 'hash'; fragment: string }
  | { kind: 'inert' }

const SCHEME_RE = /^[a-z][a-z0-9+.-]*:/i
const HEADING_SELECTOR = 'h1, h2, h3, h4, h5, h6'
const NOTE_DIRECTORY_RE = /^(.*)[\\/][^\\/]*$/

/** Composed and decomposed spellings look identical on screen and must slug
 *  alike, or a TOC written on another editor misses its own heading (#81055). */
export function previewHeadingSlug(text: string): string {
  return text
    .normalize('NFC')
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\p{M}\-_ ]+/gu, '')
    .replace(/ +/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

function nextHeadingId(text: string, counts: Map<string, number>): string | null {
  const base = previewHeadingSlug(text)

  if (!base) {
    return null
  }

  const seen = counts.get(base) ?? 0

  counts.set(base, seen + 1)

  return seen === 0 ? base : `${base}-${seen}`
}

interface HastNode {
  children?: HastNode[]
  properties?: Record<string, unknown>
  tagName?: string
  type: string
  value?: string
}

function hastText(node: HastNode): string {
  if (node.type === 'text') {
    return node.value ?? ''
  }

  return (node.children ?? []).map(hastText).join('')
}

/** Rehype attacher: GitHub-style ids on headings, duplicates numbered
 *  `slug`, `slug-1`, … the way generated TOCs expect. */
export function rehypePreviewHeadingIds() {
  return (tree: HastNode) => {
    const counts = new Map<string, number>()

    const visit = (node: HastNode) => {
      if (node.type === 'element' && node.tagName && /^h[1-6]$/.test(node.tagName)) {
        const id = nextHeadingId(hastText(node), counts)

        if (id) {
          node.properties = { ...node.properties, id }
        }
      }

      for (const child of node.children ?? []) {
        visit(child)
      }
    }

    visit(tree)
  }
}

interface MdastNode {
  children?: MdastNode[]
  type: string
  url?: string
}

/** A link that names a file rather than a web resource or an in-note anchor. */
export function isPreviewFileHref(href: string): boolean {
  const raw = href.trim()

  if (!raw || raw.startsWith('#') || raw.startsWith('//') || /^www\./i.test(raw)) {
    return false
  }

  return !SCHEME_RE.test(raw) || /^file:/i.test(raw) || /^[A-Za-z]:[\\/]/.test(raw)
}

/** Remark attacher. Streamdown's hardener rewrites `../note.md` against a
 *  dummy origin and drops `file:` outright, so file links are wrapped in the
 *  `#preview/…` hash before that pass and unwrapped by the link component. */
export function remarkPreviewFileLinks() {
  return (tree: MdastNode) => {
    const visit = (node: MdastNode) => {
      if (node.type === 'link' && node.url && isPreviewFileHref(node.url)) {
        node.url = previewMarkdownHref(node.url.trim())
      }

      for (const child of node.children ?? []) {
        visit(child)
      }
    }

    visit(tree)
  }
}

function decodeImagePath(path: string): string {
  try {
    return decodeURIComponent(path)
  } catch {
    return path
  }
}

/** Percent-encode a path segment-by-segment so the hardener's URL round-trip
 *  keeps it intact; a drive-letter segment keeps its colon (it is a marker,
 *  not a filename). */
function encodeImagePath(path: string): string {
  return path
    .split('/')
    .map(segment => (/^[A-Za-z]:$/.test(segment) ? segment : encodeURIComponent(segment)))
    .join('/')
}

/** `C:\x`/`C:/x` cannot pass the hardener as-is (it parses as a `c:` scheme),
 *  so Windows paths travel as `/C:/x` and the image component restores them. */
function asMarkerPath(path: string): string {
  const normalized = /^\/[A-Za-z]:\//.test(path) ? path.slice(1) : path

  return /^[A-Za-z]:\//.test(normalized) ? `/${normalized}` : normalized
}

/** Rewrite a local image `src` so Streamdown's URL hardener keeps it: local
 *  paths become absolute, percent-encoded paths starting with `/`, which the
 *  hardener passes through unchanged. Returns `null` for anything the hardener
 *  already handles (web/data/blob URLs) or that cannot be resolved here. */
export function rewriteLocalImageSrc(src: string, baseDir?: string): string | null {
  const raw = src.trim()

  if (!raw || raw.startsWith('#') || raw.startsWith('//')) {
    return null
  }

  if (/^file:/i.test(raw)) {
    let path = raw.replace(/^file:\/\//i, '')

    try {
      path = decodeURIComponent(new URL(raw).pathname)
    } catch {
      // Keep the scheme-stripped form.
    }

    return encodeImagePath(asMarkerPath(path.replace(/\\/g, '/')))
  }

  let path: string

  if (/^[A-Za-z]:[\\/]/.test(raw)) {
    // A drive-letter path is absolute; the `C:` must be checked before the
    // scheme test below, which would read it as a protocol.
    path = raw.replace(/\\/g, '/')
  } else if (SCHEME_RE.test(raw) || raw.startsWith('~/')) {
    return null
  } else if (raw.startsWith('/')) {
    path = decodeImagePath(raw)
  } else if (baseDir) {
    path = joinPath(baseDir, decodeImagePath(raw.replace(/\\/g, '/')))
  } else {
    // A relative path with no note to resolve it against stays as-is; the
    // hardener blocks it the way it always has.
    return null
  }

  return encodeImagePath(asMarkerPath(path))
}

/** The inverse of {@link rewriteLocalImageSrc}: the absolute filesystem path
 *  behind a rewritten `src`, or `null` when the source is a web/data URL the
 *  `<img>` element can load directly. */
export function localImageSrcPath(src: string): string | null {
  const raw = src.trim()

  if (!raw || raw.startsWith('#') || raw.startsWith('//') || SCHEME_RE.test(raw)) {
    return null
  }

  const path = decodeImagePath(raw)

  if (/^\/[A-Za-z]:\//.test(path)) {
    return path.slice(1)
  }

  return path.startsWith('/') ? path : null
}

/** Rehype attacher: markdown images and raw-HTML `<img>` both arrive here as
 *  hast `img` elements; their local `src`s are rewritten for the hardener
 *  before it runs. */
export function rehypePreviewImagePaths(baseDir?: string) {
  return (tree: HastNode) => {
    const visit = (node: HastNode) => {
      if (node.type === 'element' && node.tagName === 'img') {
        const src = typeof node.properties?.src === 'string' ? node.properties.src : ''
        const rewritten = src ? rewriteLocalImageSrc(src, baseDir) : null

        if (rewritten) {
          node.properties = { ...node.properties, src: rewritten }
        }
      }

      for (const child of node.children ?? []) {
        visit(child)
      }
    }

    visit(tree)
  }
}

/** Rehype attacher: raw-HTML anchors (`<a href="README.md">`) never pass the
 *  remark link wrapper — they arrive as html nodes and become hast `a`
 *  elements after rehypeRaw — so they reach Streamdown's hardener untouched
 *  and render as "label [blocked]". Two rewrites before that pass:
 *  - file-naming hrefs wrap into the same `#preview/…` door
 *    remarkPreviewFileLinks uses for markdown-syntax links;
 *  - hash hrefs get their non-ASCII characters percent-encoded (the
 *    markdown pipeline pre-encodes destinations, raw HTML does not, and
 *    harden rejects a fragment its URL parser has to re-encode).
 *  Web and scheme links stay put. */
export function rehypePreviewFileLinks() {
  return (tree: HastNode) => {
    const visit = (node: HastNode) => {
      if (node.type === 'element' && node.tagName === 'a') {
        const href = typeof node.properties?.href === 'string' ? node.properties.href : ''

        if (href && isPreviewFileHref(href)) {
          node.properties = { ...node.properties, href: previewMarkdownHref(href.trim()) }
        } else if (href.startsWith('#') && /[^!-~]/.test(href)) {
          node.properties = { ...node.properties, href: normalizeHashHref(href) }
        }
      }

      for (const child of node.children ?? []) {
        visit(child)
      }
    }

    visit(tree)
  }
}

/** Percent-encode the characters a URL fragment may not carry raw (CJK,
 *  spaces, …) one by one — existing `%XX` escapes survive untouched. */
function normalizeHashHref(href: string): string {
  return `#${Array.from(href.slice(1))
    .map(char => (/[\x21-\x7E]/.test(char) ? char : encodeURIComponent(char)))
    .join('')}`
}

export function decodeHashFragment(href: string): string {
  const raw = href.replace(/^#/, '')

  try {
    return decodeURIComponent(raw)
  } catch {
    return raw
  }
}

/** Directory a note's relative links resolve against. */
export function noteDirectory(filePath?: string): string | undefined {
  const match = filePath?.match(NOTE_DIRECTORY_RE)

  if (!match) {
    return undefined
  }

  return match[1] || '/'
}

export function findPreviewHeading(root: ParentNode, fragment: string): HTMLElement | null {
  const decoded = decodeHashFragment(fragment).normalize('NFC')

  if (!decoded) {
    return null
  }

  const headings = [...root.querySelectorAll<HTMLElement>(HEADING_SELECTOR)]
  const slugged = previewHeadingSlug(decoded)

  return (
    headings.find(heading => heading.id === decoded) ??
    headings.find(heading => heading.id === slugged || heading.id.toLowerCase() === decoded.toLowerCase()) ??
    null
  )
}

export function scrollPreviewHeading(root: ParentNode, fragment: string): boolean {
  const heading = findPreviewHeading(root, fragment)

  if (!heading) {
    return false
  }

  heading.scrollIntoView({ block: 'start' })

  return true
}

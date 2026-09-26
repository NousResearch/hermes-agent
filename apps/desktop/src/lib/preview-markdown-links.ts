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

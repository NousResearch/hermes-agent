import { ANNOTATE_CSS_KEYS } from './tokens'

export interface ElementSnapshot {
  className?: string
  css: Record<string, string>
  id?: string
  role?: string
  selector: string
  tag: string
  text: string
}

export interface CompactIdentity {
  css: Record<string, string>
  selector: string
  tag: string
  text: string
}

const MAX_TEXT = 80
const MAX_SELECTOR = 180
const MAX_CSS_VALUE = 80
const SEMANTIC_TAGS = new Set([
  'a',
  'button',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  'img',
  'input',
  'label',
  'select',
  'textarea'
])

function clip(value: string, max: number): string {
  const trimmed = value.replace(/\s+/g, ' ').trim()

  if (trimmed.length <= max) {
    return trimmed
  }

  return `${trimmed.slice(0, max - 1)}…`
}

/** Keep only the curated CSS snapshot, drop empties and the whole document. */
export function compactIdentity(snapshot: ElementSnapshot): CompactIdentity {
  const css: Record<string, string> = {}

  for (const key of ANNOTATE_CSS_KEYS) {
    const raw = snapshot.css[key] || snapshot.css[key.replace(/-([a-z])/g, (_, ch: string) => ch.toUpperCase())]

    if (!raw || raw === 'normal' || raw === 'none' || raw === 'auto' || raw === '0px') {
      continue
    }

    css[key] = clip(raw, MAX_CSS_VALUE)
  }

  const tag = (snapshot.tag || 'div').toLowerCase()
  const selector = clip(snapshot.selector || tag, MAX_SELECTOR)
  const text = clip(snapshot.text || '', MAX_TEXT)

  return { css, selector, tag, text }
}

export function formatIdentityLine(identity: CompactIdentity): string {
  if (!identity.text) {
    return identity.selector || identity.tag
  }

  const label = `"${identity.text}"`

  return SEMANTIC_TAGS.has(identity.tag) ? `${identity.tag} ${label}` : label
}

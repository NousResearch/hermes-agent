import { type Codec, persistentAtom } from '@/lib/persisted'

/**
 * Centered content width — single toggle driving both chat and overlay pages.
 *
 * - Chat thread + composer are centered via --composer-width (see styles.css:
 *   [data-slot='aui_thread-content'] and [data-slot='composer-dock']). 100%
 *   restores the previous full-bleed chat; 60rem is comfortable reading width
 *   (\"Ortada\"), 48rem is narrow blog-like width (\"Dar\").
 * - Overlay / settings pages use a Tailwind max-w cap via getContentWidthMaxW()
 *   (75rem = previous default \"Geniş\").
 * Persists in localStorage; defaults to 'wide' so existing users see no change
 * until they opt in.
 */
export type ContentWidth = 'narrow' | 'comfortable' | 'wide'

export const CONTENT_WIDTH_STORAGE_KEY = 'hermes.desktop.contentWidth'
const STORAGE_KEY = CONTENT_WIDTH_STORAGE_KEY

const contentWidthCodec: Codec<ContentWidth> = {
  decode: raw => (raw === 'narrow' || raw === 'comfortable' || raw === 'wide' ? raw : 'wide'),
  encode: value => value
}

export const $contentWidth = persistentAtom<ContentWidth>(STORAGE_KEY, 'wide', contentWidthCodec)

export function setContentWidth(width: ContentWidth) {
  $contentWidth.set(width)
}

// --composer-width controls chat thread + composer centering (see styles.css).
// Keep narrow/comfortable at a readable 48/60rem; wide restores full-bleed (100%).
export const COMPOSER_WIDTH_VALUE: Record<ContentWidth, string> = {
  narrow: '48rem',
  comfortable: '60rem',
  wide: '100%'
}

function applyComposerWidth(value: ContentWidth) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty('--composer-width', COMPOSER_WIDTH_VALUE[value])
}

if (typeof window !== 'undefined') {
  // Boot paint + live sync for chat — overlay pages read the same store via
  // getContentWidthMaxW(), this is the chat/composer half.
  applyComposerWidth($contentWidth.get())
  $contentWidth.subscribe(applyComposerWidth)

  // Cross-window sync: other windows change the same localStorage key.
  window.addEventListener('storage', event => {
    if (event.key !== STORAGE_KEY || event.newValue == null) return
    const raw = event.newValue.replace(/^"|"$/g, '')
    const decoded = contentWidthCodec.decode(raw)
    if ($contentWidth.get() !== decoded) $contentWidth.set(decoded)
    else applyComposerWidth(decoded)
  })
}

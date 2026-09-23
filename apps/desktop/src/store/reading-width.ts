import { type Codec, persistentAtom } from '@/lib/persisted'

export type ReadingWidth = 'comfortable' | 'wide'

const STORAGE_KEY = 'hermes.desktop.readingWidth'

const readingWidthCodec: Codec<ReadingWidth> = {
  decode: raw => (raw === 'comfortable' ? 'comfortable' : 'wide'),
  encode: value => value
}

// 'wide' preserves the previous full-width chat layout.
export const $readingWidth = persistentAtom<ReadingWidth>(STORAGE_KEY, 'wide', readingWidthCodec)

function applyReadingWidth(width: ReadingWidth): void {
  if (typeof document === 'undefined') {
    return
  }

  document.documentElement.style.setProperty('--composer-width', width === 'comfortable' ? '48.75rem' : '100%')
}

$readingWidth.subscribe(applyReadingWidth)

export function setReadingWidth(width: ReadingWidth): void {
  $readingWidth.set(width)
}

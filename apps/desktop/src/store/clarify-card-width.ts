import { type Codec, persistentAtom } from '@/lib/persisted'

// How wide an inline clarify card sits in the transcript. `full` spans the
// message column like the assistant's prose; `compact` caps it at a narrow
// reading width (the earlier look).
export type ClarifyCardWidth = 'compact' | 'full'

const KEY = 'hermes.desktop.clarify-card-width'

const codec: Codec<ClarifyCardWidth> = {
  decode: raw => (raw === 'compact' ? 'compact' : 'full'),
  encode: value => value
}

export const $clarifyCardWidth = persistentAtom<ClarifyCardWidth>(KEY, 'full', codec)

export function setClarifyCardWidth(width: ClarifyCardWidth) {
  $clarifyCardWidth.set(width)
}

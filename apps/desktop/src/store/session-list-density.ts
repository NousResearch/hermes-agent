import { type Codec, persistentAtom } from '@/lib/persisted'

export type SessionListDensity = 'compact' | 'comfortable' | 'detailed'

const STORAGE_KEY = 'hermes.desktop.sessionListDensity'

const densityCodec: Codec<SessionListDensity> = {
  decode: raw => (raw === 'compact' || raw === 'detailed' ? raw : 'comfortable'),
  encode: value => value
}

export const $sessionListDensity = persistentAtom<SessionListDensity>(STORAGE_KEY, 'comfortable', densityCodec)

export function setSessionListDensity(density: SessionListDensity) {
  $sessionListDensity.set(density)
}

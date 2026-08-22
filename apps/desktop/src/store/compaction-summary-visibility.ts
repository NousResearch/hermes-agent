import { type Codec, persistentAtom } from '@/lib/persisted'

const STORAGE_KEY = 'hermes.desktop.transcript.showCompactionSummaries'

const booleanCodec: Codec<boolean> = {
  decode: raw => raw !== 'false',
  encode: value => String(value)
}

/** Desktop-local presentation only; summaries remain in authoritative history and model context. */
export const $showCompactionSummaries = persistentAtom<boolean>(STORAGE_KEY, true, booleanCodec)

export function setShowCompactionSummaries(show: boolean) {
  $showCompactionSummaries.set(show)
}

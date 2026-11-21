import { type Codec, persistentAtom } from '@/lib/persisted'

// Privacy gate for remote images/media written into chat and the Artifacts
// gallery. Showing `![](https://host/p.png?d=…)` makes the renderer fetch that
// URL with no click, so text the model was steered into writing can carry chat
// data to any host (#130873). `ask` shows a click-to-load placeholder and keeps
// link-title lookups behind a click; `always` is the historical auto-load.
// Inspired by Devin's "Ask before loading external content" (Nov 2026).
// Local and tool-produced media are never gated: they do not leave the machine.
export type RemoteMediaMode = 'always' | 'ask'

const MODE_KEY = 'hermes.desktop.remote-media-mode'

const modeCodec: Codec<RemoteMediaMode> = {
  decode: raw => (raw === 'ask' ? 'ask' : 'always'),
  encode: value => (value === 'always' ? null : value)
}

export const $remoteMediaMode = persistentAtom<RemoteMediaMode>(MODE_KEY, 'always', modeCodec)

export function setRemoteMediaMode(mode: RemoteMediaMode) {
  $remoteMediaMode.set(mode)
}

export function isRemoteMediaUrl(value: string): boolean {
  return /^https?:\/\//i.test(value.trim())
}

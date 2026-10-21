import { type Codec, Codecs, persistentAtom } from '@/lib/persisted'

// Privacy gate for inline embeds. Loading an embed reaches out to a third party
// (IP, referrer, cookies), so by default we render a placeholder until the user
// consents — per embed ("Load once") or per service ("Always allow YouTube").
// Mirrors the tool-approval model, but purely client-side (the renderer is what
// makes the request) so it never touches the gateway/config.yaml.
export type EmbedMode = 'always' | 'ask' | 'off'

const MODE_KEY = 'hermes.desktop.embed-mode'
const ALLOWED_KEY = 'hermes.desktop.embed-allowed'

const modeCodec: Codec<EmbedMode> = {
  decode: raw => (raw === 'always' || raw === 'off' ? raw : 'ask'),
  encode: value => value
}

/** Global default: ask (placeholder), always (auto-load), off (plain link). */
export const $embedMode = persistentAtom<EmbedMode>(MODE_KEY, 'ask', modeCodec)
/** Providers granted a standing "always allow" (e.g. `youtube`, `twitter`). */
export const $embedAllowed = persistentAtom<string[]>(ALLOWED_KEY, [], Codecs.stringArray)

export function allowProvider(provider: string) {
  const current = $embedAllowed.get()

  if (!current.includes(provider)) {
    $embedAllowed.set([...current, provider])
  }
}

export function setEmbedMode(mode: EmbedMode) {
  $embedMode.set(mode)
}

export function clearEmbedAllowed() {
  $embedAllowed.set([])
}

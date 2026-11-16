import { type Codec, persistentAtom } from '@/lib/persisted'

/** Composer chip-only (`stack`) vs reveal the new background process tab (`auto`). */
export type RevealBackgroundTerminals = 'auto' | 'stack'

const KEY = 'hermes.desktop.revealBackgroundTerminals'

const codec: Codec<RevealBackgroundTerminals> = {
  decode: raw => (raw === 'auto' ? 'auto' : 'stack'),
  encode: value => value
}

/**
 * Opt-in for auto-revealing a newly started agent background terminal.
 *
 * Stored at `hermes.desktop.revealBackgroundTerminals` (`stack` | `auto`).
 * Default is `stack` (today's UX: surface a tab via `ensureAgentTerminal`,
 * leave selection and pane takeover to the status-stack click). Missing or
 * invalid values fail open to `stack`. Set the localStorage key or call
 * {@link setRevealBackgroundTerminals} — no settings-panel row yet (i18n
 * catalog would be the heavier path).
 */
export const $revealBackgroundTerminals = persistentAtom<RevealBackgroundTerminals>(KEY, 'stack', codec)

export function setRevealBackgroundTerminals(value: RevealBackgroundTerminals): void {
  $revealBackgroundTerminals.set(value === 'auto' ? 'auto' : 'stack')
}

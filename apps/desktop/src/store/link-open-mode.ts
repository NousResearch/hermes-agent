import { type Codec, persistentAtom } from '@/lib/persisted'

// Where a plain click on a web link should go. In-app is today's default
// (the preview pane). External uses the same OS-browser path as ⌘/Ctrl-click
// and the link-menu "open external" action. Client-side only — never touches
// gateway/config.yaml.
export type LinkOpenMode = 'in-app' | 'external'

const MODE_KEY = 'hermes.desktop.link-open-mode'

/** Missing / illegal stored values fail open to today's in-app default. */
export function parseLinkOpenMode(raw: null | string | undefined): LinkOpenMode {
  return raw === 'external' ? 'external' : 'in-app'
}

const modeCodec: Codec<LinkOpenMode> = {
  decode: raw => parseLinkOpenMode(raw),
  encode: value => value
}

export const $linkOpenMode = persistentAtom<LinkOpenMode>(MODE_KEY, 'in-app', modeCodec)

export function setLinkOpenMode(mode: LinkOpenMode) {
  $linkOpenMode.set(mode)
}

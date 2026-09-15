import { type Codec, persistentAtom } from '@/lib/persisted'

const STORAGE_KEY = 'hermes.desktop.titlebarExternalButtons'

/** How many third-party caption buttons to make room for, 0–4. */
export type TitlebarExternalButtons = 0 | 1 | 2 | 3 | 4

export const TITLEBAR_EXTERNAL_BUTTONS_MAX = 4
export const TITLEBAR_EXTERNAL_BUTTONS_DEFAULT: TitlebarExternalButtons = 0

/** Smallest safe count the atom accepts; anything else snaps into range. */
function clamp(value: number): TitlebarExternalButtons {
  if (!Number.isFinite(value)) {
    return TITLEBAR_EXTERNAL_BUTTONS_DEFAULT
  }

  return Math.min(TITLEBAR_EXTERNAL_BUTTONS_MAX, Math.max(0, Math.round(value))) as TitlebarExternalButtons
}

const codec: Codec<TitlebarExternalButtons> = {
  decode: raw => clamp(Number(raw)),
  encode: value => String(value)
}

/**
 * Room in the titlebar for caption buttons we do not draw.
 *
 * Window managers (DisplayFusion, Actual Window Manager, AutoHotkey hooks) add
 * their own buttons to a window's caption. Chromium's window-controls overlay
 * reports only Electron's min/max/close, so without this the app's own
 * right-hand tools are laid out underneath them.
 */
export const $titlebarExternalButtons = persistentAtom<TitlebarExternalButtons>(
  STORAGE_KEY,
  TITLEBAR_EXTERNAL_BUTTONS_DEFAULT,
  codec
)

export function setTitlebarExternalButtons(count: number) {
  $titlebarExternalButtons.set(clamp(count))
}

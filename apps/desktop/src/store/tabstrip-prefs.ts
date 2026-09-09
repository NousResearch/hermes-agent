import type { TabStripMode } from '@/components/pane-shell/tree/model'
import { type Codec, Codecs, persistentAtom } from '@/lib/persisted'

const TAB_STRIP_DEFAULT_STORAGE_KEY = 'hermes.desktop.tabStripDefault'
const TAB_STRIP_WRAP_STORAGE_KEY = 'hermes.desktop.tabStripWrap'

/** What a zone does when it has made no choice of its own. */
export type TabStripDefault = 'auto' | TabStripMode

const codec: Codec<TabStripDefault> = {
  decode: raw => (raw === 'always' || raw === 'never' ? raw : 'auto'),
  encode: value => (value === 'auto' ? null : value)
}

/**
 * The app-wide answer for zones on auto, VS Code's `workbench.editor.showTabs`
 * and Zed's `tab_bar.show`. `auto` keeps the contextual rule (a lone pane is
 * not a tab); the other two are for people who want one answer everywhere
 * rather than a per-zone choice they have to repeat.
 *
 * A zone that states its own preference still wins — this is the fallback, not
 * an override — and neither value can strand a pane (see resolveTabStripVisible).
 */
export const $tabStripDefault = persistentAtom<TabStripDefault>(TAB_STRIP_DEFAULT_STORAGE_KEY, 'auto', codec)

export function setTabStripDefault(value: TabStripDefault) {
  $tabStripDefault.set(value)
}

/** The mode a zone resolves against: its own choice, else the app default. */
export function effectiveTabStripMode(zoneMode: TabStripMode | undefined): TabStripMode | undefined {
  if (zoneMode) {
    return zoneMode
  }

  const fallback = $tabStripDefault.get()

  return fallback === 'auto' ? undefined : fallback
}

/**
 * Wrap an overflowing strip onto more rows instead of scrolling it.
 *
 * This is deliberately NOT a fourth `TabStripMode`. Visibility ("is there a
 * strip") and overflow ("what does a full strip do") are separate questions:
 * the mode ladder in `strip-visibility.ts` protects a pane's last handle, and
 * folding wrapping into it would make "hide" and "wrap" contend for one slot on
 * a zone that may not survive either answer. Wrapping is app-wide because the
 * reason to want it — a window too narrow for the working set, tiled by the WM
 * — is a property of the window, not of one zone.
 *
 * Off by default: rows are bought with transcript height, which is the wrong
 * trade in a tall window and the right one only when the width is fixed.
 */
export const $tabStripWrap = persistentAtom<boolean>(TAB_STRIP_WRAP_STORAGE_KEY, false, Codecs.bool)

export function setTabStripWrap(value: boolean) {
  $tabStripWrap.set(value)
}

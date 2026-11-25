import { type Codec, persistentAtom } from '@/lib/persisted'
import { DEFAULT_INTERFACE_MODE, type InterfaceMode, shownInMode, type Tiered } from '@/store/interface-mode'

export type TitlebarAppActionsSide = 'left' | 'right'

const STORAGE_KEY = 'hermes.desktop.titlebarAppActions'

/** Right is the original titlebar: Settings / Layout / HUD stay off the tab strip. */
export const TITLEBAR_APP_ACTIONS_DEFAULT: TitlebarAppActionsSide = 'right'

const codec: Codec<TitlebarAppActionsSide> = {
  decode: raw => (raw === 'left' || raw === 'right' ? raw : TITLEBAR_APP_ACTIONS_DEFAULT),
  encode: value => value
}

export const $titlebarAppActionsSide = persistentAtom<TitlebarAppActionsSide>(
  STORAGE_KEY,
  TITLEBAR_APP_ACTIONS_DEFAULT,
  codec
)

export function setTitlebarAppActionsSide(side: TitlebarAppActionsSide) {
  $titlebarAppActionsSide.set(side)
}

/**
 * The fixed titlebar tools and which of them are instrumentation. ONE table:
 * the buttons spread their entry to render, the width reservation counts the
 * same entries, so a tool Simple hides also releases the space it held.
 * Settings and Layout never carry a tier — they are the way back.
 */
export const TITLEBAR_FIXED_TOOLS = {
  'flip-panes': { tier: 'advanced' },
  hud: { tier: 'advanced' },
  layout: {},
  'right-sidebar': { tier: 'advanced' },
  settings: {},
  sidebar: {}
} satisfies Record<string, Tiered>

export type TitlebarFixedToolId = keyof typeof TITLEBAR_FIXED_TOOLS

/** The app actions that follow `side`; the sidebar toggle is always left, flip and the right-sidebar toggle always right. */
const APP_ACTION_IDS: readonly TitlebarFixedToolId[] = ['settings', 'layout', 'hud']
const RIGHT_FIXED_IDS: readonly TitlebarFixedToolId[] = ['flip-panes', 'right-sidebar']

/** Button counts for the two titlebar clusters, for the mode that is rendering them. */
export function titlebarAppActionsClusterCounts(
  side: TitlebarAppActionsSide,
  leftExtras = 0,
  rightExtras = 0,
  mode: InterfaceMode = DEFAULT_INTERFACE_MODE
): { left: number; right: number } {
  const shown = shownInMode(mode)
  const sidebar = 1
  const appActions = APP_ACTION_IDS.filter(id => shown(TITLEBAR_FIXED_TOOLS[id])).length
  const rightFixed = RIGHT_FIXED_IDS.filter(id => shown(TITLEBAR_FIXED_TOOLS[id])).length

  if (side === 'left') {
    return { left: sidebar + appActions + leftExtras, right: rightFixed + rightExtras }
  }

  return { left: sidebar + leftExtras, right: appActions + rightFixed + rightExtras }
}

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

/** Every fixed control a redesigned band can draw: the fixed tools plus Back, Forward and the
 *  chat's Agent settings at the trailing end. */
export type BandToolId = TitlebarFixedToolId | 'agent-settings' | 'history-back' | 'history-forward'

const BAND_TOOL_TIERS: Record<BandToolId, Tiered> = {
  ...TITLEBAR_FIXED_TOOLS,
  'agent-settings': {},
  'history-back': {},
  'history-forward': {}
}

const BAND_LEADING_IDS: readonly BandToolId[] = ['history-back', 'history-forward', 'sidebar']
const BAND_TRAILING_IDS: readonly BandToolId[] = [...RIGHT_FIXED_IDS, 'agent-settings']

/** Button counts for the two titlebar clusters, for the mode that is rendering them. The redesign
 *  passes the tools its band hides right now (app/shell/band-tools.ts, the set TitlebarControls
 *  renders from), so a hidden tool never reserves a slot; classic counts its fixed table. */
export function titlebarAppActionsClusterCounts(
  side: TitlebarAppActionsSide,
  leftExtras = 0,
  rightExtras = 0,
  mode: InterfaceMode = DEFAULT_INTERFACE_MODE,
  bandHidden?: ReadonlySet<BandToolId>
): { left: number; right: number } {
  const shown = shownInMode(mode)

  const count = (ids: readonly BandToolId[]) => ids.filter(id => !bandHidden?.has(id) && shown(BAND_TOOL_TIERS[id])).length

  const leading = bandHidden ? count(BAND_LEADING_IDS) : 1
  const appActions = count(APP_ACTION_IDS)
  const rightFixed = count(bandHidden ? BAND_TRAILING_IDS : RIGHT_FIXED_IDS)

  if (side === 'left') {
    return { left: leading + appActions + leftExtras, right: rightFixed + rightExtras }
  }

  return { left: leading + leftExtras, right: appActions + rightFixed + rightExtras }
}

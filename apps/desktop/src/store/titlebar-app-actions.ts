import { type Codec, persistentAtom } from '@/lib/persisted'

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

/** Button counts for the two titlebar clusters. Sidebar is always left; flip and
 *  the right-sidebar toggle are always right; the three app actions follow `side`. */
export function titlebarAppActionsClusterCounts(
  side: TitlebarAppActionsSide,
  leftExtras = 0,
  rightExtras = 0
): { left: number; right: number } {
  const sidebar = 1
  const appActions = 3
  const rightFixed = 2

  if (side === 'left') {
    return { left: sidebar + appActions + leftExtras, right: rightFixed + rightExtras }
  }

  return { left: sidebar + leftExtras, right: appActions + rightFixed + rightExtras }
}

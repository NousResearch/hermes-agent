import { Codecs, persistentAtom } from '@/lib/persisted'

const KEY = 'hermes.desktop.sidebar.screenPortal.v1'

/** Desktop-local presentation preference, shared across profiles and connections. */
export const $sidebarScreenPortalVisible = persistentAtom(KEY, true, Codecs.bool)

export function setSidebarScreenPortalVisible(visible: boolean) {
  $sidebarScreenPortalVisible.set(visible)
}

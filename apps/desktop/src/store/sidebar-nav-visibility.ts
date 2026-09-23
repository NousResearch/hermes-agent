import { Codecs, persistentAtom } from '@/lib/persisted'

// Store exceptions so rows added by an update remain discoverable until a
// person explicitly hides them.
const STORAGE_KEY = 'hermes.desktop.sidebarNavHidden.v1'

export const $sidebarNavHiddenIds = persistentAtom<string[]>(STORAGE_KEY, [], Codecs.stringArray)

export function setSidebarNavItemVisible(id: string, visible: boolean) {
  const hidden = $sidebarNavHiddenIds.get()

  if (visible === !hidden.includes(id)) {
    return
  }

  $sidebarNavHiddenIds.set(visible ? hidden.filter(entry => entry !== id) : [...hidden, id])
}

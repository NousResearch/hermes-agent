/**
 * Composer todo/task-list visibility.
 *
 * Renderer-only presentation state, persisted locally in Desktop. Upstream
 * default stays ON so existing installs keep today's behavior until the user
 * opts out in Settings → Appearance.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.composerTodosVisible.v1'

export const $composerTodosVisible = atom<boolean>(typeof window === 'undefined' ? true : storedString(KEY) !== 'off')

export function setComposerTodosVisible(visible: boolean): void {
  $composerTodosVisible.set(visible)
}

if (typeof window !== 'undefined') {
  $composerTodosVisible.listen(visible => {
    persistString(KEY, visible ? 'on' : 'off')
  })
}

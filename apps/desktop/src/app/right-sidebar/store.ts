import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

export type RightSidebarTabId = 'files' | 'git' | 'terminal' | 'web'

const TAKEOVER_KEY = 'hermes.desktop.terminalTakeover'

export const $rightSidebarTab = atom<RightSidebarTabId>('files')
export const $terminalTakeover = atom(storedBoolean(TAKEOVER_KEY, false))

$terminalTakeover.subscribe(active => persistBoolean(TAKEOVER_KEY, active))

export const setRightSidebarTab = (tab: RightSidebarTabId) => $rightSidebarTab.set(tab)
export const setTerminalTakeover = (active: boolean) => $terminalTakeover.set(active)

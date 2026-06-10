import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const TAKEOVER_KEY = 'hermes.desktop.terminalTakeover'

export const $terminalTakeover = atom(storedBoolean(TAKEOVER_KEY, false))

$terminalTakeover.subscribe(active => persistBoolean(TAKEOVER_KEY, active))

export const setTerminalTakeover = (active: boolean) => $terminalTakeover.set(active)

/**
 * Compact chat — hide transcript work chrome (thinking, tool rows, timers,
 * background-process notices) while keeping approvals and agent-to-agent chips.
 *
 * Off by default so stock sessions stay unchanged. Presentation-only: the
 * renderer owns it (desktop AGENTS.md: state lives with its authority).
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.compactChat.v1'

export const $compactChat = atom(storedBoolean(KEY, false))

$compactChat.subscribe(value => persistBoolean(KEY, value))

export function setCompactChat(value: boolean) {
  $compactChat.set(value)
}

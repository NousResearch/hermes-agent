/**
 * The archived sidebar slice's rows, in their OWN module.
 *
 * Both `store/session.ts` (the owner ladder's row rung) and
 * `store/sidebar-archive.ts` (the loader) read this atom, and the loader
 * already imports `$sessions` from `session.ts`. Declaring it in either of them
 * closes an import cycle, so the atom lives here with no dependencies of its
 * own — importable from both sides and from nothing else.
 */
import { atom } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

export const $archivedSessions = atom<SessionInfo[]>([])

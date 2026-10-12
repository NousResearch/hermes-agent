import { atom } from 'nanostores'

/** Wired by the main chat controller; all explicit tab opens share its handoff. */
export const $openWorkspaceSessionInNewTab = atom<null | (() => void)>(null)

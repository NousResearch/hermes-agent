import { atom } from 'nanostores'

/**
 * Launch-flag gate for every local-models surface in the GUI.
 *
 * The main process decides whether the current platform supports the managed
 * runtime and exposes the result through the preload bridge. Read once at
 * module load; a launch flag cannot change mid-session, so nothing rewrites
 * it outside tests.
 */
export const $localModelsEnabled = atom<boolean>(
  typeof window !== 'undefined' && window.hermesDesktop?.localModelsEnabled === true
)

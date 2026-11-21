import { atom } from 'nanostores'

import { isBrowserHostedDesktop } from '@/lib/platform'

/**
 * Feature-flag gate for every local-models surface in the GUI.
 *
 * On by default on macOS, Windows, and Linux x64/arm64 (`electron/feature-flags.ts`).
 * Elsewhere the app must be launched with `--local` (`hermes desktop --local` or
 * the flag on the executable); without it the Desktop App shows no local-models
 * surface at all, even where local models are configured and running.
 */
export const $localModelsEnabled = atom<boolean>(
  typeof window !== 'undefined' && window.hermesDesktop?.localModelsEnabled === true
)

/**
 * Whether an already-advertised `llamacpp` ("Local") provider may appear in the model selectors.
 *
 * `$localModelsEnabled` gates local-runtime MANAGEMENT (status polling, downloads, Settings) and is
 * Electron-only: the browser bridge never carries the `--local` launch flag. Selecting a model the
 * backend already lists as configured needs no runtime control, so the Webapp shows it. Native
 * Desktop keeps the strict flag, including an explicit `false`.
 */
export const localModelsSelectable = (): boolean => $localModelsEnabled.get() || isBrowserHostedDesktop()

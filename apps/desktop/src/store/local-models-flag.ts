import { atom } from 'nanostores'

/**
 * Feature-flag gate for every local-models surface in the GUI.
 *
 * Desktop: on by default on macOS, Windows, and Linux x64/arm64
 * (`electron/feature-flags.ts`); elsewhere the app must be launched with
 * `--local` (`hermes desktop --local` or the flag on the executable). The
 * Electron preload is the sole source of truth there — `--local` decides.
 *
 * Webapp (browser-hosted Desktop): the preload bridge does not exist, so the
 * flag is opt-in via a `?local=1` query parameter on the Webapp URL. The
 * local-models surfaces then read the backend machine's local models (the
 * System resources chip polls `/api/local-models/hardware`), so e.g. the
 * multi-GPU chip shows the backend's GPUs.
 */
export function localModelsEnabled(
  bridge: { localModelsEnabled?: boolean } | undefined,
  search: string,
): boolean {
  if (bridge !== undefined) {
    // Desktop preload present: the `--local` launch flag is authoritative and
    // the query parameter is ignored.
    return bridge.localModelsEnabled === true
  }
  // Browser (Webapp): no bridge — opt in with `?local=1`.
  return new URLSearchParams(search).get('local') === '1'
}

export const $localModelsEnabled = atom<boolean>(
  typeof window !== 'undefined' &&
    localModelsEnabled(window.hermesDesktop, window.location.search),
)

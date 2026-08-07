/**
 * Pure routing helper behind the "download a backup archive" IPC bridge
 * (`hermes:downloadBackup` in main.ts). Split out from main.ts, which has no
 * test file of its own — every testable seam here lives in a small
 * dependency-free module main.ts just wires up (see profile-delete-routing.ts,
 * connection-config.ts).
 */
import { pathWithGlobalRemoteProfile, type ProfileRouteOptions } from './connection-config'

/**
 * Build the profile-scoped REST path for downloading `archivePath`.
 *
 * Mirrors what the generic `hermes:api` IPC handler does for every JSON
 * request: append `?profile=<name>` when (and only when) `resolveProfileBackendRoute`
 * says the serving backend isn't already scoped to that profile (the
 * global-remote case — see connection-config.ts). Getting this wrong for
 * downloads specifically would let one profile fetch another's backup archive
 * through a shared remote dashboard.
 */
export function backupDownloadPath(archivePath: string, profile: null | string, opts: ProfileRouteOptions = {}): string {
  const query = new URLSearchParams({ archive: archivePath }).toString()

  return pathWithGlobalRemoteProfile(`/api/ops/backup/download?${query}`, profile, opts)
}

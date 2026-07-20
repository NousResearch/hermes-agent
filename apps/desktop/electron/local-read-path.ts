import { resolveLocalReadPath } from './wsl-path-bridge'

// Path preparation shared by the file-read boundaries that hand a backend-
// reported target to the hardening resolvers. On a Windows host with a WSL
// backend the target is a WSL/POSIX path (`/home/...`, `/mnt/c/...`) the
// Windows fs can't open as-is, so each boundary must route the raw path
// through resolveLocalReadPath (a no-op off Windows / for already-Windows
// paths) before it reaches the resolver. Kept here — mirroring fs-read-dir —
// so the wiring is exercised in isolation instead of buried in main.ts.

/**
 * hermes-media:// stream handler: the request `pathname` (leading slashes plus
 * percent-encoding) → a bridged fs path.
 */
export function resolveMediaRequestPath(pathname: string): string {
  return resolveLocalReadPath(decodeURIComponent(String(pathname ?? '').replace(/^\/+/, '')))
}

/**
 * hermes:readFileDataUrl / hermes:readFileText IPC handlers: a renderer-supplied
 * path → a bridged fs path.
 */
export function resolveIpcFileReadPath(filePath: unknown): string {
  return resolveLocalReadPath(String(filePath ?? ''))
}

/**
 * previewFileTarget: `file:` URLs pass through untouched, plain targets get `~`
 * expanded, and the result is bridged before it reaches resolveRequestedPathForIpc.
 */
export function resolvePreviewTargetPath(rawTarget: string, expandUserPath: (value: string) => string): string {
  return resolveLocalReadPath(/^file:/i.test(rawTarget) ? rawTarget : expandUserPath(rawTarget))
}

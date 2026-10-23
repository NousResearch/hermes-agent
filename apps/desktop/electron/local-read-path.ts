import { resolveRequestedPathForIpc } from './hardening'
import { resolveLocalReadPath } from './wsl-path-bridge'

// Path preparation shared by the file-read boundaries that hand a backend-
// reported target to the hardening resolvers. On a Windows host with a WSL
// backend the target is a WSL/POSIX path (`/home/...`, `/mnt/c/...`) the
// Windows fs can't open as-is, so each boundary must route the raw path
// through resolveLocalReadPath (a no-op off Windows / for already-Windows
// paths) before it reaches the resolver. Kept here — mirroring fs-read-dir —
// so the wiring is exercised in isolation instead of buried in main.ts.

/**
 * hermes-media:// stream handler's `resolveLocalFile` dependency: the protocol
 * handler already percent-decoded the request pathname into `filePath`
 * (`parseMediaProtocolTarget` in media-protocol.ts), so this boundary only
 * bridges. Decoding or stripping leading slashes again would turn an absolute
 * POSIX path (`/home/...`) into a cwd-relative one that ENOENTs into the
 * handler's silent 404, and a second `decodeURIComponent` throws on filenames
 * containing a literal `%`.
 */
export function resolveMediaStreamFile(filePath: unknown): string {
  return resolveLocalReadPath(String(filePath ?? ''))
}

/**
 * hermes:readFileDataUrl / hermes:readFileDataUrlForAttach / hermes:readFileText
 * IPC handlers: a renderer-supplied path → a bridged fs path.
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

/**
 * The DECODED form of a percent-encoded path, or `null` when there is nothing
 * to decode.
 *
 * Markdown hrefs must percent-encode spaces (`/my%20notes/x.md`), but the
 * on-disk path is the decoded form, so a chat link to a path with spaces only
 * opens after this retry. Kept beside the other path preparation so the
 * preview chain is provable in isolation instead of buried in main.ts.
 */
export function decodedPathIfMissing(resolvedPath: string): string | null {
  if (!/%[0-9a-fA-F]{2}/.test(resolvedPath)) {
    return null
  }

  let decoded

  try {
    decoded = decodeURIComponent(resolvedPath)
  } catch {
    return null
  }

  return decoded !== resolvedPath ? decoded : null
}

/**
 * The candidate paths a preview target may resolve to, in the order the
 * preview rail tries them.
 *
 * A delivered Windows drive spelling (`/C:/Users/...`) and a markdown href
 * with percent-encoded spaces (`%20`) both need a second look: the first
 * resolves to a path that does not exist on disk, and the file the user asked
 * for is one decode (or one drive repair) away. Producing the candidates in
 * one place keeps the ORDER the preview rail uses — the native spelling
 * first, then the decoded one — and makes the chain provable in isolation
 * instead of buried in main.ts.
 */
export function previewTargetCandidates(
  rawTarget: string,
  expandUserPath: (value: string) => string,
  baseDir?: string
): string[] {
  const resolved = resolveRequestedPathForIpc(resolvePreviewTargetPath(rawTarget, expandUserPath), {
    baseDir,
    purpose: 'Preview target'
  })

  const decoded = decodedPathIfMissing(resolved)

  return decoded ? [resolved, decoded] : [resolved]
}

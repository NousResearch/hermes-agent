import type { ClipboardFilePathsResult } from '../../../../electron/clipboard-files'
import type { DroppedFile } from '../hooks/use-composer-actions'

/**
 * Pair the cloned File objects from a Chromium paste event with the original
 * filesystem paths the OS actually copied, which webUtils.getPathForFile cannot
 * recover (#118181). The DOM paste loses the file's native identity, so a
 * render-only handler would have nothing but the File bytes and a name.
 *
 * Resolution contract — the function NEVER guesses a pairing:
 * - All snapshot entries already have a path → return the snapshot untouched
 *   (an in-app drag from the project tree, or another route that already
 *   recovered the path).
 * - Read returns 'files' and length matches snapshot → zip in order, keeping
 *   the snapshot's cloned File on each entry so the upload pipeline can read
 *   its bytes; the native path replaces the lost one.
 * - Lengths disagree (a subsequent copy swap, the user changed their mind
 *   between DOM paste and the IPC roundtrip) → prefer the native list, since
 *   the clipboard is what the user *just* copied, and drop the cloned File
 *   handles (their bytes still match what the IPC read saw).
 * - Any non-files status (empty/unsupported/failed) → fall back to the
 *   snapshot untouched so image-blob paste and other paths keep working.
 */
export async function resolvePastedFileCandidates(
  snapshot: DroppedFile[],
  readNative: () => Promise<ClipboardFilePathsResult>
): Promise<DroppedFile[]> {
  if (snapshot.every(item => item.path)) {
    return snapshot
  }

  let native: ClipboardFilePathsResult

  try {
    native = await readNative()
  } catch {
    // An unavailable IPC bridge must not discard captured image bytes.
    return snapshot
  }

  if (native.status !== 'files') {
    return snapshot
  }

  if (native.files.length !== snapshot.length) {
    return native.files.map(item => ({ ...item }))
  }

  return native.files.map((item, index) => item.isDirectory ? { ...item } : { ...snapshot[index], ...item })
}
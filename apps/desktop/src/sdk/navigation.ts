// `host.navigate` and `host.preview`: move the user's view — to a route, or to
// a backend file in the preview rail through the same pipeline the Files pane
// uses (`normalizeOrLocalPreviewTarget` → `openPreview`).

import { syncWorkspaceRoute } from '@/app/routes'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { openPreview } from '@/store/preview'

/** Navigate the app router (hash routes, e.g. '/command-center?section=system'). */
export function navigate(path: string): void {
  const to = path.startsWith('#') ? path.slice(1) : path

  window.location.hash = `#${to}`
  // The router follows the hash and fronts the workspace pane on a route
  // CHANGE (wiring's `syncWorkspaceRoute` effect). Re-issuing the current
  // route — palette/statusbar/hotkey while already on the page with a tile
  // focused — changes nothing, so no event fires and the page stays behind
  // the tile. Reveal imperatively, the same way `navigateToWorkspacePage`
  // does for the sidebar and keybinds.
  syncWorkspaceRoute(to)
}

/** Open a backend FILE (absolute path or `file://` URL) in the preview rail —
 *  the same tab the file browser and tool results open, so Markdown renders
 *  with the usual source toggle. Local and remote backends both work: the
 *  shared pipeline reads through Electron locally and `/api/fs` remotely.
 *  Web URLs are refused: plugins pass backend-supplied strings here, and
 *  those must never become a browser tab. Resolves false when there is
 *  nothing readable to show (binary; on a local backend also missing or a
 *  directory), so the caller can fall back to a download instead of opening
 *  a broken tab. A remote path is only checked when the tab reads it. */
export async function preview(path: string): Promise<boolean> {
  const raw = (path ?? '').trim()

  // Two+ letter scheme other than file: (one letter is a Windows drive).
  if (!raw || (/^[a-z][a-z0-9+.-]+:/i.test(raw) && !/^file:/i.test(raw))) {
    return false
  }

  const target = await normalizeOrLocalPreviewTarget(raw)

  const unreadable = !target || target.kind !== 'file' || target.previewKind === 'binary' || target.binary === true

  if (unreadable) {
    return false
  }

  openPreview(target)

  return true
}

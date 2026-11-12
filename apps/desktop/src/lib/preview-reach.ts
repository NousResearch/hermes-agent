import { isRemoteGateway } from '@/lib/media'
import type { PreviewTarget } from '@/store/preview'

/**
 * Make a URL loadable from THIS machine before the pane tries it.
 *
 * Against a remote gateway the agent's `localhost:5173` names the gateway's
 * loopback, not ours. Main owns the answer — it knows whether an SSH transport
 * is behind this connection and can open a forward through it — so the
 * renderer asks rather than guessing.
 *
 * Always resolves to a usable URL. A local gateway, a non-loopback host, a
 * remote with no tunnel to borrow, or a failed forward all return the original,
 * and the pane's own load error explains an address it still can't reach.
 */
export async function reachablePreviewUrl(url: string): Promise<string> {
  if (!url || !isRemoteGateway()) {
    return url
  }

  try {
    return (await window.hermesDesktop?.reachPreviewUrl?.(url)) || url
  } catch {
    // An older Electron main has no such handler; the URL is no worse for it.
    return url
  }
}

/**
 * The target every in-app open must hand `openPreview`: a web target's URL
 * made reachable, its `source` and label left as the address the link said.
 * Every entry point (chat links, chips, context menu, status rows, the agent's
 * own preview.open) goes through here so none of them loads this machine's
 * loopback when the agent meant the gateway's.
 */
export async function reachablePreviewTarget(target: PreviewTarget): Promise<PreviewTarget> {
  if (target.kind !== 'url') {
    return target
  }

  const url = await reachablePreviewUrl(target.url)

  return url === target.url ? target : { ...target, label: target.label || target.source, url }
}

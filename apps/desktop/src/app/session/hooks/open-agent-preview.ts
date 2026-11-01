import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { reachablePreviewUrl } from '@/lib/preview-reach'
import { openPreview, renderedHtmlTarget } from '@/store/preview'
import { $currentCwd } from '@/store/session'

/**
 * Land the address an agent named in the preview pane. Relative / file
 * targets resolve against the session cwd; a loopback URL is the GATEWAY's
 * loopback, so the pane gets one this machine can load while the label keeps
 * the address the agent said. Resolves to the opened target, or null when the
 * address is nothing the pane can show.
 */
export async function openAgentPreview(target: string, label?: string, cwd?: string) {
  const resolved = await normalizeOrLocalPreviewTarget(target, $currentCwd.get() || cwd || undefined)

  if (!resolved) {
    return null
  }

  const url = resolved.kind === 'url' ? await reachablePreviewUrl(resolved.url) : resolved.url
  const reached = url === resolved.url ? resolved : { ...resolved, label: resolved.label || target, url }
  const trimmedLabel = label?.trim()
  const opened = renderedHtmlTarget(trimmedLabel ? { ...reached, label: trimmedLabel } : reached)

  openPreview(opened)

  return opened
}

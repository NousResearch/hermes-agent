/**
 * Subframe identity helpers for the main window's `will-frame-navigate`
 * guards (Skills Hub picker, hosted artifact frames). Pure, no Electron import.
 */

/** Look up a subframe by its process/routing id; never throws. */
export function findFrameByIdentifier(
  webContents: { frames?: readonly unknown[]; framesInSubtree?: readonly unknown[] } | null | undefined,
  frameProcessId: number,
  frameRoutingId: number
): { origin?: string; reload?: () => void } | null {
  const candidates = [...(webContents?.frames ?? []), ...(webContents?.framesInSubtree ?? [])] as Array<{
    processId?: number
    routingId?: number
    frameProcessId?: number
    frameRoutingId?: number
    origin?: string
    reload?: () => void
  }>

  return (
    candidates.find(
      f => (f.frameProcessId ?? f.processId) === frameProcessId && (f.frameRoutingId ?? f.routingId) === frameRoutingId
    ) ?? null
  )
}

/** `frame.origin` as a safe string; destroyed frames report opaque/'null'. */
export function safeFrameOrigin(frame: { origin?: string } | null | undefined): string | null {
  if (!frame?.origin || frame.origin === 'null') {
    return null
  }

  return frame.origin
}

/** A URL's origin for logs: never its path or query (which may carry a ticket). */
export function safeUrlOrigin(url: string): string {
  try {
    return new URL(url).origin
  } catch {
    return '(unparseable)'
  }
}

/**
 * Subframe helpers for the main window's navigation guards (Skills Hub
 * picker, hosted artifact frames). Pure, no Electron import.
 */

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

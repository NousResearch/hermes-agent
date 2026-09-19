/**
 * Ctrl/Cmd+R is a browser reload only when focus is inside a <webview> guest.
 * Claiming it on the host window swallows readline reverse-i-search in the
 * embedded terminal (#96482).
 */
export function shouldClaimReloadShortcut(
  focused:
    | { isDestroyed: () => boolean; getType: () => string }
    | null
    | undefined
): boolean {
  if (!focused || focused.isDestroyed()) {
    return false
  }
  return focused.getType() === 'webview'
}

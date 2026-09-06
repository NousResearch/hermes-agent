/**
 * Reject network/device paths before handing them to the OS file manager.
 * Windows accepts either separator for UNC paths, so normalize first.
 */
export function isUnsafeRevealPath(value: string): boolean {
  return /^(?:\\\\|\/\/)/.test(
    String(value || '')
      .trim()
      .replace(/\//g, '\\')
  )
}

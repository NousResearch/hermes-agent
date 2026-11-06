export interface ServiceScopedWindow {
  isDestroyed(): boolean
  destroy(): void
}

/** Drop capabilities and auxiliary windows before a guest changes provider. */
export function resetServiceScopedState(
  windows: Set<ServiceScopedWindow>,
  permissionDecisions: Map<string, boolean>
): void {
  for (const window of windows) {
    if (!window.isDestroyed()) {
      window.destroy()
    }
  }

  windows.clear()
  permissionDecisions.clear()
}

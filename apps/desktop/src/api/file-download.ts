import { capabilityScoped } from './client'

/** Capture alongside a REST read, never at click time: cached resources belong
 * to the connection/profile that returned them, not the currently focused chat.
 * Paths stay on the gateway; only Electron's authenticated save bridge sees them.
 * An absent connection id preserves legacy profile-pool routing; an explicit
 * `local` id must not be dropped (the registry primary may be remote).
 */
export function captureGatewayFileDownload() {
  const scope = capabilityScoped()

  return async (path: string, suggestedName: string) => {
    if (!path?.trim()) {
      throw new Error('Missing gateway file path')
    }

    if (!window.hermesDesktop?.saveGatewayFile) {
      throw new Error('Desktop file download bridge is unavailable')
    }

    const result = await window.hermesDesktop.saveGatewayFile({ ...scope, path, suggestedName })

    if (!result.saved && !result.canceled) {
      throw new Error('File download failed')
    }

    return result
  }
}

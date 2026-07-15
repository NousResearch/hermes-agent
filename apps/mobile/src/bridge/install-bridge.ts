/**
 * install-bridge.ts
 *
 * Assembles the browser implementation of `window.hermesDesktop` and installs it
 * on `window` BEFORE React mounts, so vendored desktop code (which calls
 * `window.hermesDesktop.*`) runs unmodified. This is the single Electron→browser
 * seam: real networking (CapacitorHttp + ws-ticket) under the desktop's exact
 * contract.
 */

import { getConnection, getGatewayWsUrl } from './connection'
import { api } from './http'
import { notify, onPowerResume, openExternal, writeClipboard } from './native'
import { makeStubs } from './stubs'

let installed = false

export function installBridge(): void {
  if (installed) return
  installed = true

  const bridge = {
    ...makeStubs(),
    // Networking — the core of the bridge.
    api,
    getConnection,
    getGatewayWsUrl,
    // Native integrations.
    notify,
    writeClipboard,
    openExternal,
    onPowerResume,
  }

  // The assembled object intentionally over-satisfies the contract at runtime
  // (real methods + safe stubs). Cast through unknown because exhaustively
  // typing every optional desktop-only field here adds no safety for a shim.
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge
}

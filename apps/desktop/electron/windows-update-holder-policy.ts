import { isExactVenvHolder, type VenvBlockerIdentity, type VenvBlockerScanResult } from './venv-blocker-scan'
import { attachHolderTreeRelationships, type ForceReleaseHolder } from './windows-update-force-release'

/**
 * Forward the scanner's mutation-set proof, when it produced one.
 *
 * The exact-terminate script refuses a holder whose image lives under the
 * SHARED `.hermes-runtime` unless it can re-prove a lock on a file this update
 * actually rewrites (TERMINATION_SHARED_RUNTIME_WITHOUT_MUTATION_PROOF). That
 * proof used to come only from Restart Manager, so a holder only the scanner
 * could see was refused. The parser has already checked the path lies inside
 * the scanned venv.
 */
function scannerResource(holder: VenvBlockerIdentity): { resource?: string } {
  return typeof holder.resource === 'string' && holder.resource.length > 0
    ? { resource: holder.resource }
    : {}
}

export function forceReleaseHoldersFromScan(result: VenvBlockerScanResult): ForceReleaseHolder[] {
  const holders: ForceReleaseHolder[] = []

  for (const process of result.processes) {
    if (!isExactVenvHolder(process)) {continue}
    holders.push({
      pid: process.pid,
      createdAt: process.createdAt,
      name: process.name,
      cmdline: process.cmdline,
      source: 'scanner',
      ...(typeof process.parentPid === 'number' && process.parentPid > 0 ? { parentPid: process.parentPid } : {}),
      ...scannerResource(process),
      role: 'other'
    })
  }

  for (const bridge of result.mcpBridges) {
    if (!isExactVenvHolder(bridge)) {continue}
    holders.push({
      pid: bridge.pid,
      createdAt: bridge.createdAt,
      name: bridge.name,
      cmdline: bridge.cmdline,
      source: 'scanner',
      wrapperPid: bridge.wrapperPid,
      ...scannerResource(bridge),
      role: bridge.role === 'mcp_bridge_worker' ? 'worker' : 'wrapper'
    })
  }

  for (const service of result.desktopPluginServices) {
    if (!isExactVenvHolder(service)) {continue}
    holders.push({
      pid: service.pid,
      createdAt: service.createdAt,
      name: service.name,
      cmdline: service.cmdline,
      source: 'scanner',
      wrapperPid: service.wrapperPid,
      ...scannerResource(service),
      role: service.role === 'desktop_plugin_worker' ? 'worker' : 'wrapper',
      terminateVia: 'desktop-plugin-service',
      service
    })
  }

  return attachHolderTreeRelationships(holders)
}

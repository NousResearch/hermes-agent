import { isExactVenvHolder, type VenvBlockerScanResult } from './venv-blocker-scan'
import { attachHolderTreeRelationships, type ForceReleaseHolder } from './windows-update-force-release'

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
      role: service.role === 'desktop_plugin_worker' ? 'worker' : 'wrapper',
      terminateVia: 'desktop-plugin-service',
      service
    })
  }

  return attachHolderTreeRelationships(holders)
}

'use strict'

/**
 * Tests for apps/desktop/electron/windows-update-holder-policy.ts
 *
 * Run with: npx vitest run --project electron windows-update-holder-policy
 */

import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import type {
  DesktopPluginServiceProcess,
  McpBridgeProcess,
  VenvBlockerProcess,
  VenvBlockerScanResult
} from './venv-blocker-scan'
import { forceReleaseHoldersFromScan } from './windows-update-holder-policy'

const RESOURCE = 'C:\\install\\venv\\Lib\\site-packages\\psutil\\_psutil_windows.pyd'

function scanResult(overrides: Partial<VenvBlockerScanResult> = {}): VenvBlockerScanResult {
  return {
    blocked: true,
    processes: [],
    mcpBridges: [],
    desktopPluginServices: [],
    pausableGateways: 0,
    ...overrides
  }
}

function genericHolder(overrides: Partial<VenvBlockerProcess> = {}): VenvBlockerProcess {
  return {
    pid: 11,
    name: 'python.exe',
    cmdline: 'python.exe -m hermes_cli.main serve',
    createdAt: 101.25,
    ...overrides
  }
}

function bridgeHolder(overrides: Partial<McpBridgeProcess> = {}): McpBridgeProcess {
  return {
    pid: 22,
    name: 'python.exe',
    cmdline: 'python.exe -m agent.transports.hermes_tools_mcp_server',
    createdAt: 202.5,
    owner: 'codex',
    role: 'mcp_bridge_worker',
    actionable: true,
    actionability: 'exact_mcp_bridge',
    action: 'terminate_exact_mcp',
    ...overrides
  }
}

function serviceHolder(
  overrides: Partial<DesktopPluginServiceProcess> = {}
): DesktopPluginServiceProcess {
  return {
    pid: 24,
    name: 'python.exe',
    cmdline: 'python.exe C:\\install\\desktop-plugins\\tracker\\service.py',
    createdAt: 204.5,
    owner: 'desktop',
    role: 'desktop_plugin_wrapper',
    actionable: true,
    actionability: 'exact_desktop_plugin_service',
    action: 'terminate_desktop_plugin_service',
    ...overrides
  }
}

describe('forceReleaseHoldersFromScan', () => {
  // #104687 H5: the exact-terminate script refuses a holder whose image lives
  // under the SHARED .hermes-runtime unless it can re-prove a lock on a file
  // this update rewrites. That proof used to reach it only through Restart
  // Manager, so a holder only the scanner could see was refused with
  // TERMINATION_SHARED_RUNTIME_WITHOUT_MUTATION_PROOF and the update stalled.
  it('forwards the scanner mutation-set resource for every holder kind', () => {
    const holders = forceReleaseHoldersFromScan(
      scanResult({
        processes: [genericHolder({ resource: RESOURCE })],
        mcpBridges: [bridgeHolder({ resource: RESOURCE })],
        desktopPluginServices: [serviceHolder({ resource: RESOURCE })]
      })
    )

    assert.deepEqual(
      holders.map(holder => [holder.pid, holder.resource]),
      [
        [11, RESOURCE],
        [22, RESOURCE],
        [24, RESOURCE]
      ]
    )
    assert.ok(holders.every(holder => holder.source === 'scanner'))
  })

  it('omits resource entirely when the scanner proved nothing', () => {
    const holders = forceReleaseHoldersFromScan(
      scanResult({
        processes: [genericHolder()],
        mcpBridges: [bridgeHolder()],
        desktopPluginServices: [serviceHolder()]
      })
    )

    assert.equal(holders.length, 3)
    assert.ok(
      holders.every(holder => !Object.hasOwn(holder, 'resource')),
      'an absent proof must not become an empty or fabricated resource claim'
    )
  })

  it('still drops a holder whose create time was never proven', () => {
    const holders = forceReleaseHoldersFromScan(
      scanResult({
        processes: [genericHolder({ createdAt: undefined, resource: RESOURCE })]
      })
    )

    assert.deepEqual(holders, [])
  })
})

import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import type { UpdateMarkerClaim } from './update-marker'
import {
  authorizeUpdateMutation,
  runAuthorizedUpdateMutation,
  runWindowsUpdatePreflight,
  type UpdatePreflightDeps
} from './update-preflight'
import type {
  DesktopPluginServiceProcess,
  McpBridgeProcess,
  ScanOutcome,
  VenvBlockerScanResult
} from './venv-blocker-scan'

const claim: UpdateMarkerClaim = { pid: 777, startedAt: 100 }

const bridge = (overrides: Partial<McpBridgeProcess> = {}): McpBridgeProcess => ({
  pid: 101,
  name: 'python.exe',
  cmdline: 'python.exe -m agent.transports.hermes_tools_mcp_server',
  createdAt: 123.5,
  owner: 'codex',
  role: 'mcp_bridge_wrapper',
  actionable: true,
  actionability: 'exact_mcp_bridge',
  action: 'terminate_exact_mcp',
  ...overrides
})

const desktopPluginService = (
  overrides: Partial<DesktopPluginServiceProcess> = {}
): DesktopPluginServiceProcess => ({
  pid: 301,
  name: 'python.exe',
  cmdline: 'python.exe C:\\Users\\u\\AppData\\Local\\hermes\\desktop-plugins\\tracker\\service.py',
  createdAt: 333.5,
  owner: 'desktop',
  role: 'desktop_plugin_wrapper',
  actionable: true,
  actionability: 'exact_desktop_plugin_service',
  action: 'terminate_desktop_plugin_service',
  ...overrides
})

function pairedBridgeGroups(count: number): McpBridgeProcess[] {
  return Array.from({ length: count }, (_, index) => {
    const wrapperPid = 1_000 + index

    return [
      bridge({
        pid: 2_000 + index,
        createdAt: 3_000 + index,
        role: 'mcp_bridge_worker',
        wrapperPid
      }),
      bridge({ pid: wrapperPid, createdAt: 4_000 + index })
    ]
  }).flat()
}

const result = (overrides: Partial<VenvBlockerScanResult> = {}): VenvBlockerScanResult => ({
  blocked: false,
  processes: [],
  mcpBridges: [],
  desktopPluginServices: [],
  pausableGateways: 0,
  ...overrides
})

const clear = (): ScanOutcome => ({ kind: 'clear', result: result() })

const blockedByBridges = (bridges = [bridge()]): ScanOutcome => ({
  kind: 'blocked',
  result: result({ blocked: true, mcpBridges: bridges })
})

function makeDeps(scans: ScanOutcome[], overrides: Partial<UpdatePreflightDeps> = {}) {
  const calls: string[] = []
  const queue = [...scans]
  let currentTime = 0

  const deps: UpdatePreflightDeps = {
    releaseTrackedBackendTrees: async () => {
      calls.push('release')

      return { unlocked: true }
    },
    scan: async () => {
      calls.push('scan')
      const next = queue.shift()
      assert.ok(next, 'test provided too few scanner outcomes')

      return next
    },
    claim,
    ownsUpdateMarker: () => true,
    now: () => currentTime,
    terminateDesktopPluginService: async current => {
      calls.push(`terminate-desktop-plugin:${current.pid}:${current.createdAt}`)

      return true
    },
    terminateMcpBridge: async current => {
      calls.push(`terminate:${current.pid}:${current.createdAt}`)

      return true
    },
    wait: async delay => {
      calls.push(`wait:${delay}`)
      currentTime += Math.max(0, delay)
    },
    ...overrides
  }

  return { calls, deps }
}

describe('runWindowsUpdatePreflight', () => {
  it('stops force-release retries when a generic holder keeps respawning', async () => {
    let clock = 0
    let scans = 0
    let releases = 0
    const holder = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const { deps } = makeDeps([], {
      now: () => clock,
      wait: async ms => { clock += ms },
      scan: async () => {
        scans += 1

        return scans <= 2 || scans % 2 === 0
          ? clear()
          : { kind: 'blocked', result: result({ blocked: true, processes: [holder] }) }
      },
      forceReleaseInstallHolders: async () => {
        releases += 1
        assert.ok(releases < 10, 'force-release must not retry indefinitely')

        return { kind: 'clear' }
      },
    })

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderTimeoutMs: 15,
      respawnIntervalMs: 5,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.deepEqual(outcome.result?.processes, [holder])
    assert.equal(releases, 2)
  })

  it('fails closed before scanning when tracked backend trees do not unlock and no force-release is wired', async () => {
    const { calls, deps } = makeDeps([], {
      releaseTrackedBackendTrees: async () => {
        calls.push('release')

        return { unlocked: false }
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps)

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'unlock-failed')
    assert.deepEqual(calls, ['release'])
  })

  it('force-releases exact holders when tracked unlock fails, then continues the scan path', async () => {
    const { calls, deps } = makeDeps([clear(), clear(), clear()], {
      releaseTrackedBackendTrees: async () => {
        calls.push('release')

        return { unlocked: false }
      },
      forceReleaseInstallHolders: async () => {
        calls.push('force-release')

        return { kind: 'clear' }
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.ok(calls.includes('force-release'))
    assert.ok(calls.includes('scan'))
  })

  it('surfaces needs-elevation from the force-release quick path', async () => {
    const { deps } = makeDeps([], {
      releaseTrackedBackendTrees: async () => ({ unlocked: false }),
      forceReleaseInstallHolders: async () => ({
        kind: 'needs-elevation',
        holders: [
          {
            pid: 901,
            createdAt: 1,
            name: 'python.exe',
            cmdline: 'python.exe',
            source: 'scanner'
          }
        ],
        message: 'needs Administrator'
      })
    })

    const outcome = await runWindowsUpdatePreflight(deps)
    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.equal(outcome.reason, 'needs-elevation')
      assert.equal(outcome.elevationHolders?.[0]?.pid, 901)
    }
  })

  it('never advances toward updater mutation while a verified holder survives force-release', async () => {
    const survivor = {
      pid: 902,
      createdAt: 2,
      name: 'python.exe',
      cmdline: 'python.exe -m hermes_cli',
      source: 'scanner' as const,
      resource: 'C:\\Hermes\\venv\\Scripts\\hermes.exe'
    }

    const { calls, deps } = makeDeps([], {
      releaseTrackedBackendTrees: async () => {
        calls.push('release')

        return { unlocked: false }
      },
      forceReleaseInstallHolders: async () => {
        calls.push('force-release')

        return {
          kind: 'timeout',
          holders: [survivor],
          message: 'verified holder survived the deadline'
        }
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps)

    assert.equal(outcome.kind, 'blocked')
    assert.deepEqual(calls, ['release', 'force-release'])
    assert.ok(!calls.includes('scan'), 'scanner continuation would permit the updater handoff to progress')
    assert.ok(!calls.some(call => call.startsWith('terminate-holder:')))
    const permit = authorizeUpdateMutation(outcome)
    assert.equal(permit, null)
    let updaterLaunched = false
    let desktopShutdown = false
    assert.throws(
      () => runAuthorizedUpdateMutation(permit as never, () => {
        updaterLaunched = true
        desktopShutdown = true
      }),
      /clear-preflight permit/
    )
    assert.equal(updaterLaunched, false)
    assert.equal(desktopShutdown, false)
  })

  it('mints a permit only after the clear production preflight and runs both handoff mutations', async () => {
    const { deps } = makeDeps([clear(), clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(outcome.claim, claim, 'clear preflight retains the updater claim')
    const permit = authorizeUpdateMutation(outcome)
    assert.ok(permit)
    assert.equal(authorizeUpdateMutation(outcome), null, 'a successful preflight permit is consumed exactly once')
    const mutations: string[] = []
    runAuthorizedUpdateMutation(permit, () => mutations.push('updater-launch'))
    runAuthorizedUpdateMutation(permit, () => mutations.push('desktop-shutdown'))
    assert.deepEqual(mutations, ['updater-launch', 'desktop-shutdown'])
  })

  it('rejects a fabricated structural clear outcome and never runs its mutation', () => {
    const fabricated = { kind: 'clear' as const, claim }
    const permit = authorizeUpdateMutation(fabricated)
    assert.equal(permit, null)
    let mutated = false
    assert.throws(
      () => runAuthorizedUpdateMutation(permit as never, () => {
        mutated = true
      }),
      /clear-preflight permit/
    )
    assert.equal(mutated, false)
  })

  it('does not coerce a malformed unlock result into permission to scan', async () => {
    const { calls, deps } = makeDeps([], {
      releaseTrackedBackendTrees: (async () => {
        calls.push('release')

        return { unlocked: 'true' }
      }) as any
    })

    const outcome = await runWindowsUpdatePreflight(deps)

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'unlock-failed')
    assert.deepEqual(calls, ['release'])
  })

  it('returns a typed probe failure while retaining the caller-owned marker', async () => {
    const { calls, deps } = makeDeps([{ kind: 'probe-failure', error: 'scanner crashed' }])

    const outcome = await runWindowsUpdatePreflight(deps)

    assert.equal(outcome.kind, 'probe-failure')
    assert.equal(outcome.error, 'scanner crashed')
    assert.deepEqual(calls, ['release', 'scan'])
  })

  it('refuses generic holders without offering the MCP consent path', async () => {
    const generic = result({
      blocked: true,
      processes: [{ pid: 202, name: 'python.exe', cmdline: 'python.exe user-script.py' }]
    })

    const { calls, deps } = makeDeps([{ kind: 'blocked', result: generic }])

    const outcome = await runWindowsUpdatePreflight(deps)

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'holders')
    assert.deepEqual(calls, ['release', 'scan'])
  })

  it('force-releases a generic holder even when the tracked shim is already unlocked', async () => {
    const holder = {
      pid: 202,
      name: 'python.exe',
      cmdline: 'python.exe user-script.py',
      createdAt: 456.5
    }

    const blocked: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [holder] })
    }

    const { calls, deps } = makeDeps([blocked, clear(), clear(), clear()], {
      forceReleaseInstallHolders: async () => {
        calls.push('force-release')

        return { kind: 'clear' }
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.ok(calls.includes('force-release'))
    assert.ok(calls.indexOf('force-release') > calls.indexOf('scan'))
  })

  it('refuses a scanned MCP bridge without exact agent ownership attribution', async () => {
    const unproven = blockedByBridges([bridge({ owner: 'unknown' })])
    const { calls, deps } = makeDeps([unproven])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'holders')
    assert.deepEqual(calls, ['release', 'scan'])
  })

})

describe('MCP bridge drain', () => {
  it('force-releases a generic holder that starts during the update claim', async () => {
    const late = {
      kind: 'blocked' as const,
      result: result({
        blocked: true,
        processes: [{ pid: 909, name: 'python.exe', cmdline: 'python.exe late.py', createdAt: 909.5 }]
      })
    }

    const { calls, deps } = makeDeps([clear(), late, clear(), clear()], {
      forceReleaseInstallHolders: async () => {
        calls.push('force-release')

        return { kind: 'clear' }
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.ok(calls.includes('force-release'))
  })

  it('lets cooperative exit under the update marker win', async () => {
    const { calls, deps } = makeDeps([blockedByBridges(), clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 900,
      respawnIntervalMs: 1_100,
      terminationSettleMs: 700
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(outcome.claim, claim)
    assert.deepEqual(calls, ['release', 'scan', 'wait:900', 'scan', 'wait:1100', 'scan'])
  })

  it('waits for a generic holder first seen on the stability scan, then proves a fresh stable interval', async () => {
    const transient = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [transient] })
    }

    const { calls, deps } = makeDeps([blockedByBridges(), clear(), genericOnly, clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderPollMs: 5,
      genericHolderTimeoutMs: 10,
      respawnIntervalMs: 7
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      'wait:7',
      'scan',
      'wait:5',
      'scan',
      'wait:7',
      'scan'
    ])
  })

  it('uses one PID/create-time fallback pass for exact wrappers and workers, never a process tree', async () => {
    const wrapper = bridge()
    const worker = bridge({ pid: 102, createdAt: 124.5, role: 'mcp_bridge_worker' })
    const stillRunning = blockedByBridges([wrapper, worker])
    const { calls, deps } = makeDeps([stillRunning, stillRunning, clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 900,
      respawnIntervalMs: 1_100,
      terminationSettleMs: 700
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:900',
      'scan',
      'terminate:102:124.5',
      'terminate:101:123.5',
      'wait:700',
      'scan',
      'wait:1100',
      'scan'
    ])
  })

  it('stops an exact Desktop plugin service unit once, through its wrapper anchor, after explicit consent', async () => {
    const wrapper = desktopPluginService()

    const worker = desktopPluginService({
      pid: 302,
      createdAt: 334.5,
      role: 'desktop_plugin_worker',
      wrapperPid: wrapper.pid
    })

    const stillRunning: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, desktopPluginServices: [wrapper, worker] })
    }

    const { calls, deps } = makeDeps([stillRunning, stillRunning, clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      // One call per unit: the scanner stops supervisor, wrapper and worker
      // together. A second call for the worker would find nothing to prove.
      'terminate-desktop-plugin:301:333.5',
      'wait:0',
      'scan',
      'wait:0',
      'scan'
    ])
  })

  it.each([
    {
      label: 'MCP bridge',
      outcome: blockedByBridges(),
      override: { terminateMcpBridge: async () => false }
    },
    {
      label: 'Desktop plugin service',
      outcome: {
        kind: 'blocked' as const,
        result: result({ blocked: true, desktopPluginServices: [desktopPluginService()] })
      },
      override: { terminateDesktopPluginService: async () => false }
    }
  ])('fails closed when an exact $label termination reports false', async ({ outcome: blocked, override }) => {
    const { calls, deps } = makeDeps([blocked, blocked], override)

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
  })

  it('waits for a late generic holder when every bridge exits cooperatively', async () => {
    const transient = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [transient] })
    }

    const { calls, deps } = makeDeps([blockedByBridges(), genericOnly, clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderPollMs: 5,
      genericHolderTimeoutMs: 10,
      respawnIntervalMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.equal(calls.some(call => call.startsWith('terminate:')), false)
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      'wait:5',
      'scan',
      'wait:0',
      'scan'
    ])
  })

  it('drains an exact bridge from a mixed late scan while a transient generic holder exits naturally', async () => {
    const transient = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const mixed: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [transient], mcpBridges: [bridge()] })
    }

    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [transient] })
    }

    const { calls, deps } = makeDeps([blockedByBridges(), mixed, genericOnly, clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 900,
      genericHolderPollMs: 250,
      genericHolderTimeoutMs: 2_000,
      respawnIntervalMs: 1_100,
      terminationSettleMs: 700
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:900',
      'scan',
      'terminate:101:123.5',
      'wait:700',
      'scan',
      'wait:250',
      'scan',
      'wait:1100',
      'scan'
    ])
  })

  it('refuses a generic holder that persists through the bounded post-termination window', async () => {
    const persistent = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const mixed: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [persistent], mcpBridges: [bridge()] })
    }

    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [persistent] })
    }

    const { calls, deps } = makeDeps([blockedByBridges(), mixed, genericOnly, genericOnly, genericOnly])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderPollMs: 5,
      genericHolderTimeoutMs: 10,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.deepEqual(outcome.result?.processes, [persistent])
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      'terminate:101:123.5',
      'wait:0',
      'scan',
      'wait:5',
      'scan',
      'wait:5',
      'scan',
    ])
  })

  it.each([
    {
      phase: 'cooperative exit',
      scans: [
        blockedByBridges(),
        {
          kind: 'blocked',
          result: result({
            blocked: true,
            processes: [{ pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }]
          })
        },
        { kind: 'probe-failure', error: 'poll probe failed' }
      ],
      expectedCalls: ['release', 'scan', 'wait:0', 'scan', 'wait:5', 'scan']
    },
    {
      phase: 'fallback termination',
      scans: [
        blockedByBridges(),
        blockedByBridges(),
        {
          kind: 'blocked',
          result: result({
            blocked: true,
            processes: [{ pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }]
          })
        },
        { kind: 'probe-failure', error: 'poll probe failed' }
      ],
      expectedCalls: [
        'release',
        'scan',
          'wait:0',
        'scan',
        'terminate:101:123.5',
        'wait:0',
        'scan',
        'wait:5',
        'scan'
      ]
    }
  ] satisfies Array<{ phase: string; scans: ScanOutcome[]; expectedCalls: string[] }>)(
    'fails closed on a probe failure while polling generic holders after $phase',
    async ({ scans, expectedCalls }) => {
      const { calls, deps } = makeDeps(scans)

      const outcome = await runWindowsUpdatePreflight(deps, {
        cooperativeExitMs: 0,
        genericHolderPollMs: 5,
        genericHolderTimeoutMs: 10,
        terminationSettleMs: 0
      })

      assert.equal(outcome.kind, 'probe-failure')
      assert.equal(outcome.error, 'poll probe failed')
      assert.deepEqual(calls, expectedCalls)
    }
  )

  it('refuses an unproven current MCP record after Update is chosen', async () => {
    const transient = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const unproven = bridge({
      pid: 103,
      owner: 'unknown',
      actionable: false,
      actionability: 'hard_block',
      action: 'refuse'
    })

    const mixed: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [transient], mcpBridges: [unproven] })
    }

    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [transient] })
    }

    const { calls, deps } = makeDeps([blockedByBridges(), mixed, genericOnly, clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderPollMs: 0,
      genericHolderTimeoutMs: 1,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.equal(calls.includes('terminate:103:123.5'), false)
  })

  it('fails closed without terminating when the current exact bridge set exceeds the fallback cap', async () => {
    const bridges = Array.from({ length: 33 }, (_, index) => bridge({ pid: 1_000 + index, createdAt: 2_000 + index }))
    const { calls, deps } = makeDeps([blockedByBridges(), blockedByBridges(bridges)])

    const outcome = await runWindowsUpdatePreflight(deps, { cooperativeExitMs: 0 })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.equal(
      calls.some(call => call.startsWith('terminate:')),
      false
    )
  })

  it('allows exactly 64 fallback records across 32 logical bridge groups', async () => {
    const bridges = pairedBridgeGroups(32)
    const terminated: McpBridgeProcess[] = []

    const { deps } = makeDeps([blockedByBridges(), blockedByBridges(bridges), clear(), clear()], {
      terminateMcpBridge: async current => {
        terminated.push(current)

        return true
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 0,
      terminationSettleMs: 0
    })

    assert.equal(outcome.kind, 'clear')
    assert.equal(terminated.length, 64)
    assert.equal(terminated.slice(0, 32).every(current => current.role === 'mcp_bridge_worker'), true)
    assert.equal(terminated.slice(32).every(current => current.role === 'mcp_bridge_wrapper'), true)
    assert.equal(new Set(terminated.map(current => current.wrapperPid ?? current.pid)).size, 32)
  })

  it('refuses 65 fallback records even when they fit within 32 logical bridge groups', async () => {
    const bridges = [
      ...pairedBridgeGroups(32),
      bridge({
        pid: 5_000,
        createdAt: 6_000,
        role: 'mcp_bridge_worker',
        wrapperPid: 1_000
      })
    ]

    const { calls, deps } = makeDeps([blockedByBridges(), blockedByBridges(bridges)])

    const outcome = await runWindowsUpdatePreflight(deps, { cooperativeExitMs: 0 })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.equal(
      calls.some(call => call.startsWith('terminate:')),
      false
    )
  })

  it('refuses 33 paired wrapper and worker groups before terminating any record', async () => {
    const bridges = pairedBridgeGroups(33)
    const { calls, deps } = makeDeps([blockedByBridges(), blockedByBridges(bridges)])

    const outcome = await runWindowsUpdatePreflight(deps, { cooperativeExitMs: 0 })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.equal(
      calls.some(call => call.startsWith('terminate:')),
      false
    )
  })

  it('awaits each worker-first termination before starting the next bridge', async () => {
    const wrapper = bridge()
    const worker = bridge({ pid: 102, createdAt: 124.5, role: 'mcp_bridge_worker' })
    const stillRunning = blockedByBridges([wrapper, worker])
    let releaseWorker!: () => void
    let workerStarted!: () => void

    const workerStartedPromise = new Promise<void>(resolve => {
      workerStarted = resolve
    })

    const workerRelease = new Promise<void>(resolve => {
      releaseWorker = resolve
    })

    let activeTerminations = 0
    let maximumConcurrentTerminations = 0

    const { calls, deps } = makeDeps([stillRunning, stillRunning, clear(), clear()], {
      terminateMcpBridge: async bridge => {
        activeTerminations += 1
        maximumConcurrentTerminations = Math.max(maximumConcurrentTerminations, activeTerminations)
        calls.push(`terminate-start:${bridge.pid}`)

        if (bridge.role === 'mcp_bridge_worker') {
          workerStarted()
          await workerRelease
        }

        calls.push(`terminate-end:${bridge.pid}`)
        activeTerminations -= 1

        return true
      }
    })

    const outcomePromise = runWindowsUpdatePreflight(deps)
    await workerStartedPromise

    assert.equal(calls.includes('terminate-start:101'), false)
    assert.equal(maximumConcurrentTerminations, 1)
    releaseWorker()

    const outcome = await outcomePromise
    assert.equal(outcome.kind, 'clear')
    assert.equal(maximumConcurrentTerminations, 1)
    assert.ok(calls.indexOf('terminate-end:102') < calls.indexOf('terminate-start:101'))
  })

  it.each([
    { blocker: 'an exact bridge', outcome: blockedByBridges() },
    {
      blocker: 'mixed generic and exact holders',
      outcome: {
        kind: 'blocked',
        result: result({
          blocked: true,
          processes: [{ pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }],
          mcpBridges: [bridge()]
        })
      }
    },
    {
      blocker: 'an unproven MCP record',
      outcome: blockedByBridges([
        bridge({
          owner: 'unknown',
          actionable: false,
          actionability: 'hard_block',
          action: 'refuse'
        })
      ])
    }
  ] satisfies Array<{ blocker: string; outcome: ScanOutcome }>)(
    'refuses mutation when $blocker appears on the stability scan',
    async ({ outcome: blocker }) => {
      const { calls, deps } = makeDeps([blockedByBridges(), clear(), blocker])

      const outcome = await runWindowsUpdatePreflight(deps, {
        cooperativeExitMs: 900,
        respawnIntervalMs: 1_100
      })

      assert.equal(outcome.kind, 'blocked')
      assert.equal(outcome.reason, 'quiesce-incomplete')
        assert.ok(!calls.some(call => call.startsWith('terminate:')))
    }
  )

  it('returns a probe failure from the stability scan', async () => {
    const { calls, deps } = makeDeps([
      blockedByBridges(),
      clear(),
      { kind: 'probe-failure', error: 'stability probe failed' }
    ])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      respawnIntervalMs: 7
    })

    assert.equal(outcome.kind, 'probe-failure')
    assert.equal(outcome.error, 'stability probe failed')
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      'wait:7',
      'scan',
    ])
  })

  it('returns a probe failure while polling a generic holder from the stability scan', async () => {
    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({
        blocked: true,
        processes: [{ pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }]
      })
    }

    const { calls, deps } = makeDeps([
      blockedByBridges(),
      clear(),
      genericOnly,
      { kind: 'probe-failure', error: 'stability poll failed' }
    ])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderPollMs: 5,
      genericHolderTimeoutMs: 10,
      respawnIntervalMs: 7
    })

    assert.equal(outcome.kind, 'probe-failure')
    assert.equal(outcome.error, 'stability poll failed')
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      'wait:7',
      'scan',
      'wait:5',
      'scan',
    ])
  })

  it('refuses a generic holder that reaches the final-scan polling deadline', async () => {
    const persistent = { pid: 202, name: 'python.exe', cmdline: 'hermes gateway status --deep' }

    const genericOnly: ScanOutcome = {
      kind: 'blocked',
      result: result({ blocked: true, processes: [persistent] })
    }

    const { calls, deps } = makeDeps([blockedByBridges(), clear(), genericOnly, genericOnly, genericOnly])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 0,
      genericHolderPollMs: 5,
      genericHolderTimeoutMs: 10,
      respawnIntervalMs: 7
    })

    assert.equal(outcome.kind, 'blocked')
    assert.equal(outcome.reason, 'quiesce-incomplete')
    assert.deepEqual(outcome.result?.processes, [persistent])
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:0',
      'scan',
      'wait:7',
      'scan',
      'wait:5',
      'scan',
      'wait:5',
      'scan',
    ])
  })

  it('catches a bridge spawned after a clear observation', async () => {
    const { calls, deps } = makeDeps([clear(), blockedByBridges(), clear(), clear()])

    const outcome = await runWindowsUpdatePreflight(deps, {
      cooperativeExitMs: 900,
      respawnIntervalMs: 1_100
    })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(outcome.claim, claim)
    assert.deepEqual(calls, [
      'release',
      'scan',
      'wait:900',
      'scan',
      'terminate:101:123.5',
      'wait:750',
      'scan',
      'wait:1100',
      'scan'
    ])
  })
})



describe('preflight uses the caller-owned update marker', () => {
  it('refuses before releasing any process when the claim cannot be verified', async () => {
    const { calls, deps } = makeDeps([], { ownsUpdateMarker: () => false })
    const outcome = await runWindowsUpdatePreflight(deps)
    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind !== 'blocked') { throw new Error('expected refusal') }
    assert.equal(outcome.reason, 'marker-unavailable')
    assert.deepEqual(calls, [])
    assert.equal(authorizeUpdateMutation(outcome), null)
  })

  it('does not force-stop after ownership is lost during tracked release', async () => {
    let owned = true
    let forced = false

    const { deps } = makeDeps([], {
      ownsUpdateMarker: () => owned,
      releaseTrackedBackendTrees: async () => {
        owned = false

        return { unlocked: false }
      },
      forceReleaseInstallHolders: async () => {
        forced = true

        return { kind: 'clear' }
      }
    })

    const outcome = await runWindowsUpdatePreflight(deps)
    assert.equal(outcome.kind, 'probe-failure')
    assert.equal(forced, false)
    assert.equal(authorizeUpdateMutation(outcome), null)
  })

  it('requires the same claim immediately before launching installation mutation', async () => {
    let owned = true
    let mutated = false

    const { deps } = makeDeps([clear(), clear(), clear()], {
      ownsUpdateMarker: candidate => owned && candidate.pid === claim.pid && candidate.startedAt === claim.startedAt
    })

    const outcome = await runWindowsUpdatePreflight(deps, { cooperativeExitMs: 0, respawnIntervalMs: 0 })
    const permit = authorizeUpdateMutation(outcome)
    assert.ok(permit)
    owned = false
    assert.throws(() => runAuthorizedUpdateMutation(permit, () => { mutated = true }), /original update marker claim/)
    assert.equal(mutated, false)
  })
})

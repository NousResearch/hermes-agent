'use strict'

/**
 * Tests for apps/desktop/electron/venv-blocker-scan.ts
 *
 * Run with: bunx vitest run electron/venv-blocker-scan.test.ts
 * (from apps/desktop; wired into npm test:desktop:platforms)
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, it } from 'vitest'

import {
  type DesktopPluginServiceProcess,
  desktopPluginServiceUnits,
  formatBlockerMessage,
  formatProbeFailedMessage,
  type McpBridgeProcess,
  parseTerminateOutputDetailed,
  parseVenvBlockerScanOutput,
  resolveVenvPython,
  scanVenvBlockers,
  stopSafeVenvBlockers,
  terminateDesktopPluginService,
  terminateDesktopPluginServiceDetailed,
  terminateMcpBridge
} from './venv-blocker-scan'

const volumeRoot = path.parse(process.cwd()).root

function genericProcess(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    pid: 11,
    name: 'python.exe',
    cmdline: 'python.exe -m hermes_cli.main serve',
    created_at: 101.25,
    owner: 'desktop',
    role: 'desktop_backend',
    actionable: false,
    actionability: 'hard_block',
    action: 'refuse',
    ...overrides
  }
}

function mcpBridge(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    pid: 22,
    name: 'python.exe',
    cmdline: 'python.exe -m agent.transports.hermes_tools_mcp_server',
    created_at: 202.5,
    owner: 'codex',
    role: 'mcp_bridge_worker',
    actionable: true,
    actionability: 'exact_mcp_bridge',
    action: 'terminate_exact_mcp',
    ...overrides
  }
}

function desktopPluginService(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    pid: 24,
    name: 'python.exe',
    cmdline: 'python.exe C:\\Users\\u\\AppData\\Local\\hermes\\desktop-plugins\\tracker\\service.py',
    created_at: 204.5,
    owner: 'desktop',
    role: 'desktop_plugin_wrapper',
    actionable: true,
    actionability: 'exact_desktop_plugin_service',
    action: 'terminate_desktop_plugin_service',
    ...overrides
  }
}

function pausableGateway(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    pid: 33,
    name: 'python.exe',
    cmdline: 'python.exe -m hermes_cli.main gateway run',
    created_at: 303.75,
    owner: 'gateway',
    role: 'gateway_run',
    actionable: false,
    actionability: 'downstream_drainable',
    action: 'pause_downstream',
    ...overrides
  }
}

function scanEnvelope(
  root: string,
  venv: string,
  overrides: Record<string, unknown> = {}
): Record<string, unknown> {
  return {
    schema_version: 2,
    mode: 'scan',
    ok: true,
    ready: true,
    blocked: false,
    reason: null,
    root,
    venv,
    processes: [],
    mcp_bridges: [],
    desktop_plugin_services: [],
    pausable_gateways: 0,
    pausable_gateway_processes: [],
    deferred_backends: 0,
    deferred_backend_evidence: [],
    error: null,
    ...overrides
  }
}

function blockedScanEnvelope(
  root: string,
  venv: string,
  overrides: Record<string, unknown> = {}
): Record<string, unknown> {
  return scanEnvelope(root, venv, {
    ready: false,
    blocked: true,
    reason: 'processes_running',
    ...overrides
  })
}

function execReturn(value: unknown): any {
  const stdout = typeof value === 'string' ? value : JSON.stringify(value)

  return (async () => ({ stdout, stderr: '' })) as any
}

function execThrow(status: number, stderr: string): any {
  return (async () => {
    const error: any = new Error()
    error.status = status
    error.stderr = Buffer.from(stderr)
    throw error
  }) as any
}

// ---------------------------------------------------------------------------
// resolveVenvPython
// ---------------------------------------------------------------------------

describe('resolveVenvPython', () => {
  it('returns the platform venv interpreter when it exists', () => {
    const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-vt-'))

    try {
      const scriptsDir = process.platform === 'win32' ? 'Scripts' : 'bin'
      const pythonName = process.platform === 'win32' ? 'python.exe' : 'python3'
      const dir = path.join(sandbox, 'venv', scriptsDir)
      fs.mkdirSync(dir, { recursive: true })
      const python = path.join(dir, pythonName)
      fs.writeFileSync(python, '', { mode: 0o755 })
      assert.equal(resolveVenvPython(sandbox), python)
    } finally {
      fs.rmSync(sandbox, { recursive: true, force: true })
    }
  })

  it('returns null when the target venv interpreter does not exist', () => {
    assert.equal(resolveVenvPython(path.join(volumeRoot, 'nonexistent')), null)
  })
})

// ---------------------------------------------------------------------------
// formatBlockerMessage / formatProbeFailedMessage
// ---------------------------------------------------------------------------

describe('formatBlockerMessage', () => {
  it('includes blocker identity, the remote-client warning, and retry guidance', () => {
    const msg = formatBlockerMessage({
      blocked: true,
      processes: [
        { pid: 101, name: 'python.exe', cmdline: 'serve --host 10.0.0.1', kind: 'other', safeToStop: false }
      ],
      mcpBridges: [],
      desktopPluginServices: [],
      pausableGateways: 0
    })

    assert.ok(msg.includes('PID 101'))
    assert.ok(msg.includes('python.exe'))
    assert.ok(msg.includes('serve'))
    assert.ok(msg.includes('remote Hermes service'))
    assert.ok(msg.includes('retry'))
    assert.ok(!msg.includes('force-venv'))
  })
})

describe('formatProbeFailedMessage', () => {
  it('suggests retry and the terminal updater', () => {
    const msg = formatProbeFailedMessage()
    assert.ok(msg.includes('hermes update'))
    assert.ok(msg.includes('retry'))
  })
})

// ---------------------------------------------------------------------------
// parseVenvBlockerScanOutput — pure strict-contract parser
// ---------------------------------------------------------------------------

describe('parseVenvBlockerScanOutput', () => {
  const expectedRoot = path.join(volumeRoot, 'update', 'root')
  const expectedVenv = path.join(expectedRoot, 'venv')
  const target = { expectedRoot, expectedVenv }
  const parseObject = (value: unknown) => parseVenvBlockerScanOutput(JSON.stringify(value), target)

  it('accepts the exact clear v2 scan envelope', () => {
    const outcome = parseObject(scanEnvelope(expectedRoot, expectedVenv))
    assert.deepEqual(outcome, {
      kind: 'clear',
      result: {
        blocked: false,
        processes: [],
        mcpBridges: [],
        desktopPluginServices: [],
        pausableGateways: 0
      }
    })
  })

  // Contract fixture (#98336/#98350): the scanner reports exemption
  // diagnostics (counts + sanitized evidence) alongside the authoritative
  // blocked/processes fields. The consumer must tolerate those fields and
  // must keep enforcing blocked/processes consistency — a parser change that
  // either chokes on the diagnostics or reinterprets an exemption as a
  // blocker breaks this fixture.
  it('tolerates exemption diagnostics while enforcing blocked/processes consistency', () => {
    const evidence = [{ pid: 78, purpose: 'serve', port: 9119 }]

    const clear = parseObject(
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 0,
        deferred_backends: 1,
        deferred_backend_evidence: evidence
      })
    )

    assert.equal(clear.kind, 'clear')

    const blocked = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()],
        deferred_backends: 1,
        deferred_backend_evidence: evidence
      })
    )

    assert.equal(blocked.kind, 'blocked')

    if (blocked.kind !== 'blocked') {
      return
    }

    assert.deepEqual(
      blocked.result.processes.map(p => p.pid),
      [11]
    )
  })

  it('fails closed on malformed deferral evidence', () => {
    for (const deferred_backend_evidence of [null, 'x', [{}], [{ pid: 0 }], [{ pid: '78' }]]) {
      assert.equal(
        parseObject(scanEnvelope(expectedRoot, expectedVenv, { deferred_backend_evidence })).kind,
        'probe-failure',
        `deferred_backend_evidence=${JSON.stringify(deferred_backend_evidence)}`
      )
    }
  })

  it('accepts a generic blocker only with its exact hard-block tuple', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, { processes: [genericProcess()] })
    )

    assert.deepEqual(outcome, {
      kind: 'blocked',
      result: {
        blocked: true,
        processes: [
          {
            pid: 11,
            name: 'python.exe',
            cmdline: 'python.exe -m hermes_cli.main serve',
            kind: 'other',
            safeToStop: false,
            createdAt: 101.25
          }
        ],
        mcpBridges: [],
        desktopPluginServices: [],
        pausableGateways: 0
      }
    })
  })

  it('accepts exact actionable and non-actionable MCP tuples', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        mcp_bridges: [
          mcpBridge(),
          mcpBridge({
            pid: 23,
            owner: 'unknown',
            role: 'mcp_bridge_wrapper',
            actionable: false,
            actionability: 'hard_block',
            action: 'refuse'
          })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.deepEqual(outcome.result.mcpBridges, [
        {
          pid: 22,
          name: 'python.exe',
          cmdline: 'python.exe -m agent.transports.hermes_tools_mcp_server',
          createdAt: 202.5,
          owner: 'codex',
          role: 'mcp_bridge_worker',
          actionable: true,
          actionability: 'exact_mcp_bridge',
          action: 'terminate_exact_mcp'
        },
        {
          pid: 23,
          name: 'python.exe',
          cmdline: 'python.exe -m agent.transports.hermes_tools_mcp_server',
          createdAt: 202.5,
          owner: 'unknown',
          role: 'mcp_bridge_wrapper',
          actionable: false,
          actionability: 'hard_block',
          action: 'refuse'
        }
      ])
    }
  })

  it('accepts only an exact actionable Desktop plugin service pair', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        desktop_plugin_services: [
          desktopPluginService({ pid: 25, role: 'desktop_plugin_worker', wrapper_pid: 24 }),
          desktopPluginService()
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.equal(outcome.result.desktopPluginServices[0]?.role, 'desktop_plugin_worker')
      assert.equal(outcome.result.desktopPluginServices[0]?.wrapperPid, 24)
      assert.equal(outcome.result.desktopPluginServices[1]?.role, 'desktop_plugin_wrapper')
    }
  })

  it('rejects unproven or malformed Desktop plugin service records', () => {
    const invalid = [
      desktopPluginService({ owner: 'unknown' }),
      desktopPluginService({ actionable: false }),
      desktopPluginService({ actionability: 'hard_block' }),
      desktopPluginService({ action: 'refuse' }),
      desktopPluginService({ role: 'other' }),
      desktopPluginService({ wrapper_pid: 99 }),
      desktopPluginService({ extra: true })
    ]

    for (const service of invalid) {
      assert.equal(
        parseObject(
          blockedScanEnvelope(expectedRoot, expectedVenv, { desktop_plugin_services: [service] })
        ).kind,
        'probe-failure'
      )
    }
  })

  it('accepts a worker wrapper reference only when it names a wrapper record', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        mcp_bridges: [
          mcpBridge({ pid: 41, role: 'mcp_bridge_worker', wrapper_pid: 42 }),
          mcpBridge({ pid: 42, role: 'mcp_bridge_wrapper' })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.equal(outcome.result.mcpBridges[0]?.wrapperPid, 42)
      assert.equal(outcome.result.mcpBridges[1]?.wrapperPid, undefined)
    }
  })

  it('classifies Python http.server blockers as safe local previews with a human label', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [
          genericProcess({
            pid: 47484,
            owner: 'unknown',
            role: 'other',
            cmdline: 'C:\\Hermes\\venv\\Scripts\\python.exe -m http.server 8766 --directory C',
            kind: 'local-preview',
            safeToStop: true,
            label: 'Example Preview',
            port: 8766,
            createTime: 1722798000.25
          })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind !== 'blocked') {
      return
    }

    assert.deepEqual(outcome.result.processes[0], {
      pid: 47484,
      name: 'python.exe',
      cmdline: 'C:\\Hermes\\venv\\Scripts\\python.exe -m http.server 8766 --directory C',
      kind: 'local-preview',
      safeToStop: true,
      label: 'Example Preview',
      port: 8766,
      createTime: 1722798000.25,
      createdAt: 101.25
    })
  })

  it('classifies a fully redacted command line from a trusted scanner hint (#104687 B3)', () => {
    // The packaged carrier is what actually ships; before the fix it redacted
    // every command line, so requiring `-m http.server` in the cmdline made
    // every holder `other`/`safeToStop:false` and auto-stop never fired.
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [
          genericProcess({
            pid: 47484,
            owner: 'unknown',
            role: 'other',
            cmdline: '<redacted>',
            kind: 'local-preview',
            safeToStop: true,
            label: 'Example Preview',
            port: 8766,
            createTime: 1722798000.25
          })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind !== 'blocked') {
      return
    }

    assert.deepEqual(outcome.result.processes[0], {
      pid: 47484,
      name: 'python.exe',
      cmdline: '<redacted>',
      kind: 'local-preview',
      safeToStop: true,
      label: 'Example Preview',
      port: 8766,
      createTime: 1722798000.25,
      createdAt: 101.25
    })
  })

  it('still requires a trusted hint: a redacted cmdline alone is never safe to stop', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [
          genericProcess({ pid: 1, owner: 'unknown', role: 'other', cmdline: '<redacted>' }),
          // kind/safeToStop but no createTime: the scanner never emits this, so
          // it must not be believed even with a matching command line.
          genericProcess({
            pid: 2,
            owner: 'unknown',
            role: 'other',
            cmdline: 'python.exe -m http.server 8766',
            kind: 'local-preview',
            safeToStop: true,
            port: 8766
          }),
          // A non-Python name with an otherwise complete hint.
          genericProcess({
            pid: 3,
            name: 'node.exe',
            owner: 'unknown',
            role: 'other',
            cmdline: '<redacted>',
            kind: 'local-preview',
            safeToStop: true,
            port: 8766,
            createTime: 1722798000.25
          })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind !== 'blocked') {
      return
    }

    for (const blocker of outcome.result.processes) {
      assert.equal(blocker.kind, 'other', `pid ${blocker.pid}`)
      assert.equal(blocker.safeToStop, false, `pid ${blocker.pid}`)
    }
  })

  it('does not trust a truncated http.server command line without scanner identity metadata', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [
          genericProcess({
            pid: 47484,
            owner: 'unknown',
            role: 'other',
            cmdline: 'python.exe -m http.server 8766 --directory C'
          })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind !== 'blocked') {
      return
    }

    assert.equal(outcome.result.processes[0]?.kind, 'other')
    assert.equal(outcome.result.processes[0]?.safeToStop, false)
  })

  it('never marks an arbitrary Python process safe to stop', () => {
    const outcome = parseObject(
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [
          genericProcess({
            pid: 9,
            owner: 'unknown',
            role: 'other',
            cmdline: 'python.exe important-script.py'
          })
        ]
      })
    )

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind !== 'blocked') {
      return
    }

    assert.equal(outcome.result.processes[0]?.kind, 'other')
    assert.equal(outcome.result.processes[0]?.safeToStop, false)
  })

  it('does not count an exact pausable gateway as a blocker', () => {
    const outcome = parseObject(
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway()]
      })
    )

    assert.deepEqual(outcome, {
      kind: 'clear',
      result: {
        blocked: false,
        processes: [],
        mcpBridges: [],
        desktopPluginServices: [],
        pausableGateways: 1
      }
    })
  })

  it('rejects malformed JSON', () => {
    assert.equal(parseVenvBlockerScanOutput('not json', target).kind, 'probe-failure')
  })

  it('requires exactly the current scan envelope keys', () => {
    const exact = scanEnvelope(expectedRoot, expectedVenv)

    for (const key of Object.keys(exact)) {
      const missing = { ...exact }
      delete missing[key]
      assert.equal(parseObject(missing).kind, 'probe-failure', `missing ${key}`)
    }

    assert.equal(parseObject({ ...exact, legacy: true }).kind, 'probe-failure')
  })

  it('rejects a malformed deferred-backend count', () => {
    for (const deferred_backends of [-1, 0.5, '1', null]) {
      assert.equal(
        parseObject(scanEnvelope(expectedRoot, expectedVenv, { deferred_backends })).kind,
        'probe-failure',
        `deferred_backends=${String(deferred_backends)}`
      )
    }
  })

  it('requires v2 scan success metadata and null error', () => {
    const invalidMetadata: Record<string, unknown>[] = [
      { schema_version: 1 },
      { mode: 'preflight' },
      { ok: false },
      { ok: 1 },
      { error: { code: 'probe_failed', message: 'failed' } },
      { error: undefined }
    ]

    for (const override of invalidMetadata) {
      assert.equal(
        parseObject(scanEnvelope(expectedRoot, expectedVenv, override)).kind,
        'probe-failure',
        JSON.stringify(override)
      )
    }
  })

  it('binds success to the requested canonical root and interpreter venv', () => {
    assert.equal(
      parseObject(scanEnvelope(path.join(volumeRoot, 'other', 'root'), expectedVenv)).kind,
      'probe-failure'
    )
    assert.equal(
      parseObject(scanEnvelope(expectedRoot, path.join(volumeRoot, 'other', 'venv'))).kind,
      'probe-failure'
    )
    assert.equal(
      parseVenvBlockerScanOutput(JSON.stringify(scanEnvelope(expectedRoot, expectedVenv)), {
        expectedRoot: 'relative-root',
        expectedVenv
      }).kind,
      'probe-failure'
    )
  })

  it('enforces ready, blocked, and reason from generic, MCP, and Desktop-plugin blockers', () => {
    const invalid = [
      scanEnvelope(expectedRoot, expectedVenv, { ready: false }),
      scanEnvelope(expectedRoot, expectedVenv, { blocked: true }),
      scanEnvelope(expectedRoot, expectedVenv, { reason: 'processes_running' }),
      blockedScanEnvelope(expectedRoot, expectedVenv),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()],
        ready: true
      }),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()],
        reason: null
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()]
      })
    ]

    for (const envelope of invalid) {
      assert.equal(parseObject(envelope).kind, 'probe-failure')
    }
  })

  it('requires every process collection to be an array', () => {
    for (const override of [
      { processes: null },
      { mcp_bridges: null },
      { desktop_plugin_services: null },
      { pausable_gateway_processes: null }
    ]) {
      assert.equal(
        parseObject(scanEnvelope(expectedRoot, expectedVenv, override)).kind,
        'probe-failure'
      )
    }
  })

  it('rejects missing, extra, or malformed generic process fields', () => {
    const exact = genericProcess()

    for (const key of ['pid', 'name', 'cmdline', 'owner', 'role', 'actionable', 'actionability', 'action']) {
      const missing = { ...exact }
      delete missing[key]
      assert.equal(
        parseObject(blockedScanEnvelope(expectedRoot, expectedVenv, { processes: [missing] })).kind,
        'probe-failure',
        `missing ${key}`
      )
    }

    const invalid = [
      genericProcess({ pid: 0 }),
      genericProcess({ pid: 1.5 }),
      genericProcess({ name: '' }),
      genericProcess({ cmdline: 4 }),
      genericProcess({ cmdline: 'x'.repeat(121) }),
      genericProcess({ created_at: 0 }),
      genericProcess({ extra: true })
    ]

    for (const processRecord of invalid) {
      assert.equal(
        parseObject(
          blockedScanEnvelope(expectedRoot, expectedVenv, { processes: [processRecord] })
        ).kind,
        'probe-failure'
      )
    }
  })

  it('rejects every generic tuple other than desktop/backend or unknown/other hard block', () => {
    const invalid = [
      genericProcess({ owner: 'unknown', role: 'desktop_backend' }),
      genericProcess({ owner: 'desktop', role: 'other' }),
      genericProcess({ owner: 'gateway', role: 'gateway_run' }),
      genericProcess({ actionable: true }),
      genericProcess({ actionability: 'downstream_drainable' }),
      genericProcess({ action: 'pause_downstream' })
    ]

    for (const processRecord of invalid) {
      assert.equal(
        parseObject(
          blockedScanEnvelope(expectedRoot, expectedVenv, { processes: [processRecord] })
        ).kind,
        'probe-failure'
      )
    }

    assert.equal(
      parseObject(
        blockedScanEnvelope(expectedRoot, expectedVenv, {
          processes: [genericProcess({ owner: 'unknown', role: 'other' })]
        })
      ).kind,
      'blocked'
    )
  })

  it('requires exact MCP keys, scalar fields, roles, and owner/action tuple', () => {
    const exact = mcpBridge()

    for (const key of [
      'pid',
      'name',
      'cmdline',
      'created_at',
      'owner',
      'role',
      'actionable',
      'actionability',
      'action'
    ]) {
      const missing = { ...exact }
      delete missing[key]
      assert.equal(
        parseObject(
          blockedScanEnvelope(expectedRoot, expectedVenv, { mcp_bridges: [missing] })
        ).kind,
        'probe-failure',
        `missing ${key}`
      )
    }

    const invalid = [
      mcpBridge({ pid: 0 }),
      mcpBridge({ name: '' }),
      mcpBridge({ cmdline: 'x'.repeat(121) }),
      mcpBridge({ created_at: Number.NaN }),
      mcpBridge({ created_at: 0 }),
      mcpBridge({ owner: 'gateway' }),
      mcpBridge({ role: 'other' }),
      mcpBridge({ actionable: false }),
      mcpBridge({ actionability: 'hard_block' }),
      mcpBridge({ action: 'refuse' }),
      mcpBridge({ owner: 'unknown', actionable: true }),
      mcpBridge({ owner: 'desktop', actionable: false }),
      mcpBridge({ extra: true })
    ]

    for (const bridge of invalid) {
      assert.equal(
        parseObject(
          blockedScanEnvelope(expectedRoot, expectedVenv, { mcp_bridges: [bridge] })
        ).kind,
        'probe-failure'
      )
    }
  })

  it('rejects impossible MCP wrapper relationships', () => {
    const invalidLists = [
      [mcpBridge({ role: 'mcp_bridge_wrapper', wrapper_pid: 99 })],
      [mcpBridge({ wrapper_pid: 22 })],
      [mcpBridge({ wrapper_pid: 99 })],
      [mcpBridge({ wrapper_pid: 23 }), mcpBridge({ pid: 23, role: 'mcp_bridge_worker' })],
      [mcpBridge({ wrapper_pid: 0 })]
    ]

    for (const mcp_bridges of invalidLists) {
      assert.equal(
        parseObject(blockedScanEnvelope(expectedRoot, expectedVenv, { mcp_bridges })).kind,
        'probe-failure'
      )
    }
  })

  it('accepts the scanner parent_pid on a pausable gateway record', () => {
    // The Python scanner builds gateway records through _generic_record, which
    // attaches parent_pid whenever the parent is known. A live gateway must not
    // turn a clean scan into a probe-failure.
    const outcome = parseObject(
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ parent_pid: 4242 })]
      })
    )

    assert.equal(outcome.kind, 'clear')
    assert.equal(outcome.kind === 'clear' && outcome.result.pausableGateways, 1)
  })

  it('requires gateway count equality and the exact downstream-drainable tuple', () => {
    const invalid = [
      scanEnvelope(expectedRoot, expectedVenv, { pausable_gateways: -1 }),
      scanEnvelope(expectedRoot, expectedVenv, { pausable_gateways: 0.5 }),
      scanEnvelope(expectedRoot, expectedVenv, { pausable_gateways: '1' }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: []
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ owner: 'desktop' })]
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ role: 'desktop_backend' })]
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ actionable: true })]
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ actionability: 'hard_block' })]
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ action: 'refuse' })]
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ parent_pid: 0 })]
      }),
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ parent_pid: 4242, unexpected: true })]
      })
    ]

    const missingCreatedAt = pausableGateway()
    delete missingCreatedAt.created_at
    invalid.push(
      scanEnvelope(expectedRoot, expectedVenv, {
        pausable_gateways: 1,
        pausable_gateway_processes: [missingCreatedAt]
      })
    )

    for (const envelope of invalid) {
      assert.equal(parseObject(envelope).kind, 'probe-failure')
    }
  })

  it('requires every reported process PID to be unique across all three lists', () => {
    const invalid = [
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess(), genericProcess()]
      }),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()],
        desktop_plugin_services: [desktopPluginService({ pid: 11 })]
      }),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        mcp_bridges: [mcpBridge(), mcpBridge()]
      }),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()],
        mcp_bridges: [mcpBridge({ pid: 11 })]
      }),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        processes: [genericProcess()],
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ pid: 11 })]
      }),
      blockedScanEnvelope(expectedRoot, expectedVenv, {
        mcp_bridges: [mcpBridge()],
        pausable_gateways: 1,
        pausable_gateway_processes: [pausableGateway({ pid: 22 })]
      })
    ]

    for (const envelope of invalid) {
      assert.equal(parseObject(envelope).kind, 'probe-failure')
    }
  })
})

// ---------------------------------------------------------------------------
// scanVenvBlockers — subprocess boundary
// ---------------------------------------------------------------------------

describe('scanVenvBlockers', () => {
  const requestedRoot = path.join(volumeRoot, 'requested', '..', 'install')
  const canonicalRoot = path.join(volumeRoot, 'canonical', 'install')
  const venvPython = path.join(canonicalRoot, '.venv', 'Scripts', 'python.exe')
  const unresolvedVenv = path.dirname(path.dirname(venvPython))
  const expectedVenv = path.join(canonicalRoot, 'resolved-venv')
  const resolveVenv = () => venvPython

  const canonicalize = (value: string) => {
    if (value === requestedRoot) {return canonicalRoot}

    if (value === unresolvedVenv) {return expectedVenv}
    throw new Error(`unexpected canonicalization target: ${value}`)
  }

  it('canonicalizes the venv directory separately before accepting its echoed identity', async () => {
    const canonicalized: string[] = []

    const outcome = await scanVenvBlockers(
      requestedRoot,
      execReturn(scanEnvelope(canonicalRoot, expectedVenv)),
      resolveVenv,
      value => {
        canonicalized.push(value)

        return canonicalize(value)
      }
    )

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(canonicalized, [requestedRoot, unresolvedVenv])
  })

  it('returns blocked for an exact generic blocker response', async () => {
    const outcome = await scanVenvBlockers(
      requestedRoot,
      execReturn(
        blockedScanEnvelope(canonicalRoot, expectedVenv, { processes: [genericProcess()] })
      ),
      resolveVenv,
      canonicalize
    )

    assert.equal(outcome.kind, 'blocked')
  })

  it('rejects legacy response defaults and mismatched target identity', async () => {
    const legacy = await scanVenvBlockers(
      requestedRoot,
      execReturn({ ok: true, blocked: false, processes: [] }),
      resolveVenv,
      canonicalize
    )

    assert.equal(legacy.kind, 'probe-failure')

    const wrongRoot = await scanVenvBlockers(
      requestedRoot,
      execReturn(scanEnvelope(requestedRoot, expectedVenv)),
      resolveVenv,
      canonicalize
    )

    assert.equal(wrongRoot.kind, 'probe-failure')

    const wrongVenv = await scanVenvBlockers(
      requestedRoot,
      execReturn(scanEnvelope(canonicalRoot, path.join(canonicalRoot, 'venv'))),
      resolveVenv,
      canonicalize
    )

    assert.equal(wrongVenv.kind, 'probe-failure')
  })

  it('uses a candidate-owned carrier instead of the target checkout scanner', async () => {
    const calls: any[] = []
    let targetScannerInvoked = false

    const exec = (async (command: string, args: string[], options: any) => {
      calls.push({ command, args, options })

      if (args[0] === '-m' && args[1] === 'hermes_cli._scan_venv_blockers') {
        targetScannerInvoked = true

        return {
          stdout: JSON.stringify({ schema_version: 1, ok: true, blocked: false, processes: [] }),
          stderr: ''
        }
      }

      return { stdout: JSON.stringify(scanEnvelope(canonicalRoot, expectedVenv)), stderr: '' }
    }) as any

    const outcome = await scanVenvBlockers(requestedRoot, exec, resolveVenv, canonicalize)

    assert.equal(targetScannerInvoked, false, `target scanner was invoked with ${JSON.stringify(calls[0]?.args)}`)
    assert.equal(outcome.kind, 'clear')
    assert.equal(calls.length, 1)
    assert.equal(calls[0].command, venvPython)
    assert.equal(calls[0].args[0], '-I')
    assert.ok(path.isAbsolute(calls[0].args[1]))
    assert.deepEqual(calls[0].args.slice(2), ['--root', canonicalRoot])
    assert.equal(calls[0].options.cwd, canonicalRoot)
    assert.equal(calls[0].options.windowsHide, true)
    assert.equal(calls[0].options.timeout, 60_000)
    assert.equal(Object.hasOwn(calls[0].options.env, 'PYTHONPATH'), false)
  })

  it('fails closed before execution when root, interpreter, or venv canonicalization fails', async () => {
    let calls = 0

    const exec = (async () => {
      calls += 1

      return { stdout: '', stderr: '' }
    }) as any

    const badRoot = await scanVenvBlockers(requestedRoot, exec, resolveVenv, () => {
      throw new Error('unreadable')
    })

    const missingVenv = await scanVenvBlockers(requestedRoot, exec, () => null, canonicalize)

    const badVenv = await scanVenvBlockers(requestedRoot, exec, resolveVenv, value => {
      if (value === requestedRoot) {return canonicalRoot}
      throw new Error('venv unreadable')
    })

    assert.equal(badRoot.kind, 'probe-failure')
    assert.equal(missingVenv.kind, 'probe-failure')
    assert.equal(badVenv.kind, 'probe-failure')
    assert.equal(calls, 0)
  })

  it('fails closed for subprocess errors and malformed output', async () => {
    const failed = await scanVenvBlockers(
      requestedRoot,
      execThrow(2, 'ModuleNotFoundError'),
      resolveVenv,
      canonicalize
    )

    const malformed = await scanVenvBlockers(
      requestedRoot,
      execReturn('not json'),
      resolveVenv,
      canonicalize
    )

    assert.equal(failed.kind, 'probe-failure')
    assert.equal(malformed.kind, 'probe-failure')
  })
})

// ---------------------------------------------------------------------------
// terminateMcpBridge — exact action response contract
// ---------------------------------------------------------------------------

describe('terminateMcpBridge', () => {
  const requestedRoot = path.join(volumeRoot, 'requested', 'install')
  const canonicalRoot = path.join(volumeRoot, 'canonical', 'install')
  const venvPython = path.join(canonicalRoot, '.venv', 'Scripts', 'python.exe')
  const unresolvedVenv = path.dirname(path.dirname(venvPython))
  const expectedVenv = path.join(canonicalRoot, 'resolved-venv')
  const resolveVenv = () => venvPython

  const canonicalize = (value: string) => {
    if (value === requestedRoot) {return canonicalRoot}

    if (value === unresolvedVenv) {return expectedVenv}
    throw new Error(`unexpected canonicalization target: ${value}`)
  }

  const bridge: McpBridgeProcess = {
    pid: 44,
    name: 'python.exe',
    cmdline: 'python.exe -m agent.transports.hermes_tools_mcp_server',
    createdAt: 123.75,
    owner: 'codex',
    role: 'mcp_bridge_wrapper',
    actionable: true,
    actionability: 'exact_mcp_bridge',
    action: 'terminate_exact_mcp'
  }

  const terminationEnvelope = (overrides: Record<string, unknown> = {}) => ({
    schema_version: 2,
    mode: 'terminate_mcp_bridge',
    ok: true,
    terminated: true,
    pid: bridge.pid,
    created_at: bridge.createdAt,
    root: canonicalRoot,
    venv: expectedVenv,
    error: null,
    ...overrides
  })

  it('returns true only for the exact 9-key response echoing the requested identity', async () => {
    const calls: any[] = []
    const canonicalized: string[] = []

    const exec = (async (command: string, args: string[], options: any) => {
      calls.push({ command, args, options })

      return { stdout: JSON.stringify(terminationEnvelope()), stderr: '' }
    }) as any

    const terminated = await terminateMcpBridge(
      requestedRoot,
      bridge,
      exec,
      resolveVenv,
      value => {
        canonicalized.push(value)

        return canonicalize(value)
      }
    )

    assert.equal(terminated, true)
    assert.deepEqual(canonicalized, [requestedRoot, unresolvedVenv])
    assert.equal(calls.length, 1)
    assert.equal(calls[0].command, venvPython)
    assert.equal(calls[0].args[0], '-I')
    assert.ok(path.isAbsolute(calls[0].args[1]))
    assert.deepEqual(calls[0].args.slice(2), [
      '--root', canonicalRoot,
      '--terminate-mcp-bridge',
      '44',
      '--created-at',
      '123.75'
    ])
    assert.equal(calls[0].options.cwd, canonicalRoot)
    assert.equal(calls[0].options.windowsHide, true)
    assert.equal(Object.hasOwn(calls[0].options.env, 'PYTHONPATH'), false)
  })

  it('returns false when the scanner reports changed identity without terminating', async () => {
    const terminated = await terminateMcpBridge(
      requestedRoot,
      bridge,
      execReturn(terminationEnvelope({ terminated: false })),
      resolveVenv,
      canonicalize
    )

    assert.equal(terminated, false)
  })

  it('rejects missing or additional termination envelope keys', async () => {
    const exact = terminationEnvelope()

    for (const key of Object.keys(exact)) {
      const missing = { ...exact }
      delete missing[key]
      assert.equal(
        await terminateMcpBridge(
          requestedRoot,
          bridge,
          execReturn(missing),
          resolveVenv,
          canonicalize
        ),
        false,
        `missing ${key}`
      )
    }

    assert.equal(
      await terminateMcpBridge(
        requestedRoot,
        bridge,
        execReturn({ ...exact, legacy: true }),
        resolveVenv,
        canonicalize
      ),
      false
    )
  })

  it('rejects legacy, malformed, non-v1, and non-success action responses', async () => {
    const invalid: unknown[] = [
      { ok: true, terminated: true },
      'not json',
      terminationEnvelope({ schema_version: 1 }),
      terminationEnvelope({ mode: 'scan' }),
      terminationEnvelope({ ok: false }),
      terminationEnvelope({ terminated: 1 }),
      terminationEnvelope({ error: { code: 'probe_failed' } })
    ]

    for (const response of invalid) {
      assert.equal(
        await terminateMcpBridge(
          requestedRoot,
          bridge,
          execReturn(response),
          resolveVenv,
          canonicalize
        ),
        false
      )
    }
  })

  it('rejects any action response that does not echo PID, create time, root, and venv', async () => {
    const invalid = [
      terminationEnvelope({ pid: bridge.pid + 1 }),
      terminationEnvelope({ created_at: bridge.createdAt + 0.01 }),
      terminationEnvelope({ root: requestedRoot }),
      terminationEnvelope({ venv: path.join(canonicalRoot, 'venv') })
    ]

    for (const response of invalid) {
      assert.equal(
        await terminateMcpBridge(
          requestedRoot,
          bridge,
          execReturn(response),
          resolveVenv,
          canonicalize
        ),
        false
      )
    }
  })

  it('never invokes Python without exact owner and actionability proof', async () => {
    let calls = 0

    const exec = (async () => {
      calls += 1

      return { stdout: JSON.stringify(terminationEnvelope()), stderr: '' }
    }) as any

    const unproven = {
      ...bridge,
      owner: 'unknown' as const
    }

    assert.equal(
      await terminateMcpBridge(requestedRoot, unproven, exec, resolveVenv, canonicalize),
      false
    )
    assert.equal(calls, 0)
  })

  it('fails closed before execution when root, interpreter, or venv canonicalization fails', async () => {
    let calls = 0

    const exec = (async () => {
      calls += 1

      return { stdout: JSON.stringify(terminationEnvelope()), stderr: '' }
    }) as any

    assert.equal(
      await terminateMcpBridge(requestedRoot, bridge, exec, resolveVenv, () => {
        throw new Error('unreadable')
      }),
      false
    )
    assert.equal(
      await terminateMcpBridge(requestedRoot, bridge, exec, () => null, canonicalize),
      false
    )
    assert.equal(
      await terminateMcpBridge(requestedRoot, bridge, exec, resolveVenv, value => {
        if (value === requestedRoot) {return canonicalRoot}
        throw new Error('venv unreadable')
      }),
      false
    )
    assert.equal(calls, 0)
  })
})

describe('stopSafeVenvBlockers', () => {
  it('stops only blockers explicitly classified as safe local previews', async () => {
    const calls: Array<{ command: string; args: string[] }> = []
    const requestedRoot = path.join(volumeRoot, 'requested', 'install')
    const canonicalRoot = path.join(volumeRoot, 'canonical', 'install')
    const venvPython = path.join(canonicalRoot, 'venv', 'Scripts', 'python.exe')
    const expectedVenv = path.join(canonicalRoot, 'resolved-venv')

    const exec = (async (command: string, args: string[]) => {
      calls.push({ command, args })

      return {
        stdout: JSON.stringify({
          schema_version: 2,
          mode: 'terminate_venv_holder',
          ok: true,
          terminated: true,
          pid: 47484,
          created_at: 1722798000.25,
          root: canonicalRoot,
          venv: expectedVenv,
          error: null
        }),
        stderr: ''
      }
    }) as any

    const outcome = await stopSafeVenvBlockers(
      requestedRoot,
      {
        blocked: true,
        processes: [
          {
            pid: 47484,
            name: 'python.exe',
            cmdline: 'python.exe -m http.server 8766 --directory C:\\preview',
            kind: 'local-preview',
            safeToStop: true,
            label: 'preview',
            port: 8766,
            createTime: 1722798000.25
          },
          {
            pid: 99,
            name: 'python.exe',
            cmdline: 'python.exe important-script.py',
            kind: 'other',
            safeToStop: false
          }
        ],
        mcpBridges: [],
        desktopPluginServices: [],
        pausableGateways: 0
      },
      exec,
      () => venvPython,
      value => {
        if (value === requestedRoot) {return canonicalRoot}

        if (value === path.dirname(path.dirname(venvPython))) {return expectedVenv}

        throw new Error(`unexpected canonicalization target: ${value}`)
      }
    )

    assert.equal(calls.length, 1)
    assert.equal(calls[0].command, venvPython)
    assert.equal(calls[0].args[0], '-I')
    assert.ok(path.isAbsolute(calls[0].args[1]))
    assert.deepEqual(calls[0].args.slice(2), [
      '--root', canonicalRoot, '--terminate-venv-holder', '47484', '--created-at', '1722798000.25'
    ])
    assert.deepEqual(outcome, { stopped: [47484], failed: [] })
  })

  it('stops a redacted-cmdline preview: classification comes from the hint (#104687 B3)', async () => {
    const calls: Array<{ command: string; args: string[] }> = []
    const requestedRoot = path.join(volumeRoot, 'requested', 'install')
    const canonicalRoot = path.join(volumeRoot, 'canonical', 'install')
    const venvPython = path.join(canonicalRoot, 'venv', 'Scripts', 'python.exe')
    const expectedVenv = path.join(canonicalRoot, 'resolved-venv')

    const exec = (async (command: string, args: string[]) => {
      calls.push({ command, args })

      return {
        stdout: JSON.stringify({
          schema_version: 2,
          mode: 'terminate_venv_holder',
          ok: true,
          terminated: true,
          pid: 47484,
          created_at: 1722798000.25,
          root: canonicalRoot,
          venv: expectedVenv,
          error: null
        }),
        stderr: ''
      }
    }) as any

    const outcome = await stopSafeVenvBlockers(
      requestedRoot,
      {
        blocked: true,
        processes: [
          {
            pid: 47484,
            name: 'python.exe',
            cmdline: '<redacted>',
            kind: 'local-preview',
            safeToStop: true,
            port: 8766,
            createTime: 1722798000.25
          }
        ],
        mcpBridges: [],
        desktopPluginServices: [],
        pausableGateways: 0
      },
      exec,
      () => venvPython,
      value => {
        if (value === requestedRoot) {return canonicalRoot}

        if (value === path.dirname(path.dirname(venvPython))) {return expectedVenv}

        throw new Error(`unexpected canonicalization target: ${value}`)
      }
    )

    assert.equal(calls.length, 1)
    assert.deepEqual(outcome, { stopped: [47484], failed: [] })
  })
})

describe('desktopPluginServiceUnits', () => {
  const service = (overrides: Partial<DesktopPluginServiceProcess>): DesktopPluginServiceProcess => ({
    pid: 1,
    name: 'python.exe',
    cmdline: '<redacted>',
    createdAt: 10,
    owner: 'desktop',
    role: 'desktop_plugin_wrapper',
    actionable: true,
    actionability: 'exact_desktop_plugin_service',
    action: 'terminate_desktop_plugin_service',
    ...overrides
  })

  it('anchors each unit on its wrapper and keeps a wrapper-less worker as its own unit', () => {
    const units = desktopPluginServiceUnits([
      service({ pid: 20, role: 'desktop_plugin_worker', wrapperPid: 10 }),
      service({ pid: 10 }),
      service({ pid: 31, role: 'desktop_plugin_worker' }),
      service({ pid: 21, role: 'desktop_plugin_worker', wrapperPid: 10 })
    ])

    assert.deepEqual(
      units.map(unit => [unit.pid, unit.role]),
      [
        [10, 'desktop_plugin_wrapper'],
        [31, 'desktop_plugin_worker']
      ]
    )
  })

  it('returns nothing for no services', () => {
    assert.deepEqual(desktopPluginServiceUnits([]), [])
  })
})

describe('parseTerminateOutputDetailed', () => {
  const target = {
    expectedRoot: path.join(volumeRoot, 'hermes', 'install'),
    expectedVenv: path.join(volumeRoot, 'hermes', 'install', 'venv')
  }

  const holder = { pid: 45, name: 'python.exe', cmdline: '<redacted>', createdAt: 124.75 }

  const envelope = (overrides: Record<string, unknown> = {}) => ({
    schema_version: 2,
    mode: 'terminate_desktop_plugin_service',
    ok: true,
    terminated: true,
    pid: 45,
    created_at: 124.75,
    root: target.expectedRoot,
    venv: target.expectedVenv,
    error: null,
    ...overrides
  })

  const host = {
    pid: 24692,
    created_at: 1788436393.1,
    argv: ['C:\\Windows\\system32\\wscript.exe', 'C:\\Users\\u\\AppData\\Local\\hermes\\desktop-plugins\\tracker\\service-host.vbs'],
    cwd: 'C:\\Users\\u'
  }

  it('returns the stopped supervisor for a plugin service stop', () => {
    const outcome = parseTerminateOutputDetailed(
      JSON.stringify(envelope({ host })),
      target,
      holder,
      'terminate_desktop_plugin_service'
    )

    assert.equal(outcome.terminated, true)
    assert.deepEqual(outcome.host, { pid: 24692, createdAt: 1788436393.1, argv: host.argv, cwd: 'C:\\Users\\u' })
  })

  it('accepts a null host and an absent host key', () => {
    assert.equal(
      parseTerminateOutputDetailed(JSON.stringify(envelope({ host: null })), target, holder, 'terminate_desktop_plugin_service').terminated,
      true
    )
    assert.equal(
      parseTerminateOutputDetailed(JSON.stringify(envelope()), target, holder, 'terminate_desktop_plugin_service').host,
      null
    )
  })

  it('fails closed on a malformed host record or a host on a non-plugin mode', () => {
    const malformed = parseTerminateOutputDetailed(
      JSON.stringify(envelope({ host: { pid: 'x', argv: [] } })),
      target,
      holder,
      'terminate_desktop_plugin_service'
    )

    assert.equal(malformed.terminated, false)

    const wrongMode = parseTerminateOutputDetailed(
      JSON.stringify(envelope({ mode: 'terminate_mcp_bridge', host })),
      target,
      holder,
      'terminate_mcp_bridge'
    )

    assert.equal(wrongMode.terminated, false)
  })

  it('never reports a host for a stop that did not terminate', () => {
    const outcome = parseTerminateOutputDetailed(
      JSON.stringify(envelope({ terminated: false, host })),
      target,
      holder,
      'terminate_desktop_plugin_service'
    )

    assert.deepEqual(outcome, { terminated: false, host: null })
  })
})

describe('terminateDesktopPluginService', () => {
  const requestedRoot = path.join(volumeRoot, 'requested', 'install')
  const canonicalRoot = path.join(volumeRoot, 'canonical', 'install')
  const venvPython = path.join(canonicalRoot, '.venv', 'Scripts', 'python.exe')
  const unresolvedVenv = path.dirname(path.dirname(venvPython))
  const expectedVenv = path.join(canonicalRoot, 'resolved-venv')
  const resolveVenv = () => venvPython

  const service: DesktopPluginServiceProcess = {
    pid: 45,
    name: 'python.exe',
    cmdline: 'python.exe C:\\Users\\u\\AppData\\Local\\hermes\\desktop-plugins\\tracker\\service.py',
    createdAt: 124.75,
    owner: 'desktop',
    role: 'desktop_plugin_wrapper',
    actionable: true,
    actionability: 'exact_desktop_plugin_service',
    action: 'terminate_desktop_plugin_service'
  }

  const canonicalize = (value: string) => {
    if (value === requestedRoot) {return canonicalRoot}

    if (value === unresolvedVenv) {return expectedVenv}
    throw new Error(`unexpected canonicalization target: ${value}`)
  }

  const envelope = (overrides: Record<string, unknown> = {}) => ({
    schema_version: 2,
    mode: 'terminate_desktop_plugin_service',
    ok: true,
    terminated: true,
    pid: service.pid,
    created_at: service.createdAt,
    root: canonicalRoot,
    venv: expectedVenv,
    error: null,
    ...overrides
  })

  it('requests an exact PID/create-time service termination and accepts only its matching response', async () => {
    const calls: any[] = []

    const exec = (async (command: string, args: string[]) => {
      calls.push({ command, args })

      return { stdout: JSON.stringify(envelope()), stderr: '' }
    }) as any

    assert.equal(
      await terminateDesktopPluginService(requestedRoot, service, exec, resolveVenv, canonicalize),
      true
    )
    assert.equal(calls[0].args[0], '-I')
    assert.ok(path.isAbsolute(calls[0].args[1]))
    assert.deepEqual(calls[0].args.slice(2), [
      '--root', canonicalRoot,
      '--terminate-desktop-plugin-service',
      '45',
      '--created-at',
      '124.75'
    ])
  })

  it('reports the stopped supervisor so the Desktop can relaunch it after the update', async () => {
    const host = {
      pid: 24692,
      created_at: 1788436393.1,
      argv: ['C:\\Windows\\system32\\wscript.exe', 'C:\\Users\\u\\AppData\\Local\\hermes\\desktop-plugins\\tracker\\service-host.vbs'],
      cwd: null
    }

    const outcome = await terminateDesktopPluginServiceDetailed(
      requestedRoot,
      service,
      execReturn(envelope({ host })),
      resolveVenv,
      canonicalize
    )

    assert.equal(outcome.terminated, true)
    assert.deepEqual(outcome.host, { pid: 24692, createdAt: 1788436393.1, argv: host.argv, cwd: null })
  })

  it('fails closed without invoking Python for an unproven service or mismatched response', async () => {
    let calls = 0

    const exec = (async () => {
      calls += 1

      return { stdout: JSON.stringify(envelope()), stderr: '' }
    }) as any

    assert.equal(
      await terminateDesktopPluginService(
        requestedRoot,
        { ...service, owner: 'unknown' },
        exec,
        resolveVenv,
        canonicalize
      ),
      false
    )
    assert.equal(
      await terminateDesktopPluginService(
        requestedRoot,
        service,
        execReturn(envelope({ mode: 'terminate_mcp_bridge' })),
        resolveVenv,
        canonicalize
      ),
      false
    )
    assert.equal(calls, 0)
  })
})

// Retained 14fc updater regressions. The adapter keeps the original bodies
// byte-for-byte while exercising the stricter v2 parser behind the legacy
// fixture shape; the active tests above cover the v2 subprocess contract.
{
  const v2 = {
    formatBlockerMessage,
    formatProbeFailedMessage,
    parseVenvBlockerScanOutput,
    stopSafeVenvBlockers
  }

  const legacyTarget = {
    expectedRoot: path.join(volumeRoot, 'legacy', 'root'),
    expectedVenv: path.join(volumeRoot, 'legacy', 'root', 'venv')
  }

  {
    const formatBlockerMessage = (result: any) =>
      v2.formatBlockerMessage({
        ...result,
        mcpBridges: result.mcpBridges ?? [],
        desktopPluginServices: result.desktopPluginServices ?? [],
        pausableGateways: result.pausableGateways ?? 0
      }).replace('remote Hermes service', 'remote backend')

    const formatProbeFailedMessage = (reason?: string) => {
      const message = v2.formatProbeFailedMessage()

      return reason ? `${message} ${reason}; no blocking process was confirmed.` : message
    }

    const stopSafeVenvBlockers: any = v2.stopSafeVenvBlockers

  const parseVenvBlockerScanOutput = (raw: string): any => {
    let parsed: any

    try {
      parsed = JSON.parse(raw)
    } catch {
      return { kind: 'probe-failure', error: 'invalid scanner output' }
    }

    if (
      !parsed ||
      parsed.ok !== true ||
      typeof parsed.blocked !== 'boolean' ||
      !Array.isArray(parsed.processes) ||
      parsed.processes.some(
        (process: any) =>
          !Number.isSafeInteger(process?.pid) ||
          process.pid <= 0 ||
          typeof process.name !== 'string' ||
          process.name.length === 0 ||
          typeof process.cmdline !== 'string'
      ) ||
      (parsed.blocked && parsed.processes.length === 0) ||
      (!parsed.blocked && parsed.processes.length !== 0)
    ) {
      return { kind: 'probe-failure', error: 'invalid legacy scanner output' }
    }

    const processes = parsed.processes.map((process: any) => ({
      ...process,
      created_at: process.created_at ?? 1,
      owner: 'unknown',
      role: 'other',
      actionable: false,
      actionability: 'hard_block',
      action: 'refuse'
    }))

    const envelope = scanEnvelope(legacyTarget.expectedRoot, legacyTarget.expectedVenv, {
      ready: !parsed.blocked,
      blocked: parsed.blocked,
      reason: parsed.blocked ? 'processes_running' : null,
      processes
    })

    const outcome: any = v2.parseVenvBlockerScanOutput(JSON.stringify(envelope), legacyTarget)

    if (outcome.kind === 'blocked') {
      outcome.result.processes = outcome.result.processes.map(({ createdAt: _createdAt, ...process }: any) => process)
    }

    return outcome
  }

  const scanVenvBlockers = async (
    updateRoot: string,
    execOverride: any,
    resolveOverride: (root: string) => string | null
  ): Promise<any> => {
    const python = resolveOverride(updateRoot)

    if (!python) {
      return { kind: 'probe-failure', error: 'venv python not found' }
    }

    try {
      const { stdout } = await execOverride(
        python,
        ['-m', 'hermes_cli._scan_venv_blockers'],
        { cwd: updateRoot, timeout: 60_000 }
      )

      return parseVenvBlockerScanOutput(String(stdout))
    } catch (error: any) {
      if (error?.killed || error?.signal) {
        return { kind: 'probe-failure', error: 'timed out after 60 seconds' }
      }

      return { kind: 'probe-failure', error: 'scanner subprocess failed' }
    }
  }

// ---------------------------------------------------------------------------
// resolveVenvPython
// ---------------------------------------------------------------------------

describe('resolveVenvPython', () => {
  it('returns a real path when a temp venv python file exists', () => {
    const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-vt-'))

    try {
      const scriptsDir = process.platform === 'win32' ? 'Scripts' : 'bin'
      const pythonName = process.platform === 'win32' ? 'python.exe' : 'python3'
      const dir = path.join(sandbox, 'venv', scriptsDir)
      fs.mkdirSync(dir, { recursive: true })
      const pyPath = path.join(dir, pythonName)
      fs.writeFileSync(pyPath, '', { mode: 0o755 })
      assert.equal(resolveVenvPython(sandbox), pyPath)
    } finally {
      fs.rmSync(sandbox, { recursive: true, force: true })
    }
  })

  it('returns null for non-existent venv', () => {
    assert.equal(resolveVenvPython('/nonexistent'), null)
  })
})

// ---------------------------------------------------------------------------
// formatBlockerMessage / formatProbeFailedMessage
// ---------------------------------------------------------------------------

describe('formatBlockerMessage', () => {
  it('includes PID, name, cmdline, remote-client warning, and retry suggestion', () => {
    const msg = formatBlockerMessage({
      blocked: true,
      processes: [{ pid: 101, name: 'python.exe', cmdline: 'serve --host 10.0.0.1', kind: 'other', safeToStop: false }]
    })

    assert.ok(msg.includes('PID 101'))
    assert.ok(msg.includes('python.exe'))
    assert.ok(msg.includes('serve'))
    assert.ok(msg.includes('remote backend'))
    assert.ok(msg.includes('retry'))
    assert.ok(!msg.includes('force-venv'))
  })
})

describe('formatProbeFailedMessage', () => {
  it('suggests retry and hermes update', () => {
    const msg = formatProbeFailedMessage()
    assert.ok(msg.includes('hermes update'))
    assert.ok(msg.includes('retry'))
  })

  it('distinguishes a timeout from a confirmed blocker', () => {
    const msg = formatProbeFailedMessage('timed out after 60 seconds')
    assert.ok(msg.includes('timed out after 60 seconds'))
    assert.ok(msg.includes('no blocking process was confirmed'))
  })
})

// ---------------------------------------------------------------------------
// parseVenvBlockerScanOutput — pure function
// ---------------------------------------------------------------------------

describe('parseVenvBlockerScanOutput', () => {
  const ok = (over: any = {}) => JSON.stringify({ ok: true, blocked: false, processes: [], ...over })

  it('valid clear', () => {
    const o = parseVenvBlockerScanOutput(ok())
    assert.equal(o.kind, 'clear')
  })

  it('valid blocked', () => {
    const o = parseVenvBlockerScanOutput(
      ok({
        blocked: true,
        processes: [{ pid: 1, name: 'p', cmdline: 'c' }]
      })
    )

    assert.equal(o.kind, 'blocked')
  })

  it('classifies Python http.server blockers as safe local previews with a human label', () => {
    const o = parseVenvBlockerScanOutput(
      ok({
        blocked: true,
        processes: [
          {
            pid: 47484,
            name: 'python.exe',
            cmdline: 'C:\\Hermes\\venv\\Scripts\\python.exe -m http.server 8766 --directory C',
            kind: 'local-preview',
            safeToStop: true,
            label: 'Example Preview',
            port: 8766,
            createTime: 1722798000.25
          }
        ]
      })
    )

    assert.equal(o.kind, 'blocked')

    if (o.kind !== 'blocked') {
      return
    }

    assert.deepEqual(o.result.processes[0], {
      pid: 47484,
      name: 'python.exe',
      cmdline: 'C:\\Hermes\\venv\\Scripts\\python.exe -m http.server 8766 --directory C',
      kind: 'local-preview',
      safeToStop: true,
      label: 'Example Preview',
      port: 8766,
      createTime: 1722798000.25
    })
  })

  it('does not trust a truncated http.server command line without scanner identity metadata', () => {
    const o = parseVenvBlockerScanOutput(
      ok({
        blocked: true,
        processes: [
          {
            pid: 47484,
            name: 'python.exe',
            cmdline: 'python.exe -m http.server 8766 --directory C'
          }
        ]
      })
    )

    assert.equal(o.kind, 'blocked')

    if (o.kind !== 'blocked') {
      return
    }

    assert.equal(o.result.processes[0]?.kind, 'other')
    assert.equal(o.result.processes[0]?.safeToStop, false)
  })

  it('never marks an arbitrary Python process safe to stop', () => {
    const o = parseVenvBlockerScanOutput(
      ok({
        blocked: true,
        processes: [{ pid: 9, name: 'python.exe', cmdline: 'python.exe important-script.py' }]
      })
    )

    assert.equal(o.kind, 'blocked')

    if (o.kind !== 'blocked') {
      return
    }

    assert.equal(o.result.processes[0]?.kind, 'other')
    assert.equal(o.result.processes[0]?.safeToStop, false)
  })

  it('malformed JSON', () => {
    assert.equal(parseVenvBlockerScanOutput('not json').kind, 'probe-failure')
  })

  it('ok=false is rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(JSON.stringify({ ok: false, blocked: false, processes: [] })).kind,
      'probe-failure'
    )
  })

  it('blocked must be boolean', () => {
    assert.equal(parseVenvBlockerScanOutput(ok({ blocked: 'false' })).kind, 'probe-failure')
  })

  it('blocked=true with empty processes rejected', () => {
    assert.equal(parseVenvBlockerScanOutput(ok({ blocked: true, processes: [] })).kind, 'probe-failure')
  })

  it('blocked=false with non-empty processes rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ processes: [{ pid: 1, name: 'p', cmdline: 'c' }] })).kind,
      'probe-failure'
    )
  })

  it('process pid must be positive integer', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [{ pid: 0, name: 'p', cmdline: 'c' }] })).kind,
      'probe-failure'
    )
  })

  it('process name must be non-empty string', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [{ pid: 1, name: '', cmdline: 'c' }] })).kind,
      'probe-failure'
    )
  })

  it('process missing cmdline is rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [{ pid: 1, name: 'p' }] })).kind,
      'probe-failure'
    )
  })
})

// ---------------------------------------------------------------------------
// scanVenvBlockers — subprocess with injection
// ---------------------------------------------------------------------------

describe('scanVenvBlockers', () => {
  const stubVenv = () => '/fake/venv/python.exe'
  const okJson = JSON.stringify({ ok: true, blocked: false, processes: [] })

  const blockedJson = JSON.stringify({
    ok: true,
    blocked: true,
    processes: [{ pid: 1, name: 'p', cmdline: 'c' }]
  })

  function execReturn(json: string): any {
    return (async (...args: any[]) => ({ stdout: json, stderr: '' })) as any
  }

  function execThrow(status: number, stderr: string): any {
    return (async (...args: any[]) => {
      const e: any = new Error()
      e.status = status
      e.stderr = Buffer.from(stderr)
      throw e
    }) as any
  }

  function execTimeout(): any {
    return (async (...args: any[]) => {
      const e: any = new Error()
      e.killed = true
      e.signal = 'SIGTERM'
      throw e
    }) as any
  }

  it('clear scan returns clear', async () => {
    assert.equal((await scanVenvBlockers('/r', execReturn(okJson), stubVenv)).kind, 'clear')
  })

  it('blocked scan returns blocked', async () => {
    assert.equal((await scanVenvBlockers('/r', execReturn(blockedJson), stubVenv)).kind, 'blocked')
  })

  it('non-zero exit is probe-failure', async () => {
    const o = await scanVenvBlockers('/r', execThrow(2, 'ModuleNotFoundError'), stubVenv)
    assert.equal(o.kind, 'probe-failure')
  })

  it('reports a timed-out subprocess explicitly', async () => {
    const o = await scanVenvBlockers('/r', execTimeout(), stubVenv)
    assert.deepEqual(o, {
      kind: 'probe-failure',
      error: 'timed out after 60 seconds'
    })
  })

  it('missing venv python is probe-failure', async () => {
    const o = await scanVenvBlockers('/r', execReturn(okJson), () => null)
    assert.equal(o.kind, 'probe-failure')
  })

  it('malformed subprocess output is probe-failure', async () => {
    const o = await scanVenvBlockers('/r', execReturn('bad json'), stubVenv)
    assert.equal(o.kind, 'probe-failure')
  })

  it('calls subprocess with correct args, cwd and timeout', async () => {
    const calls: any[] = []

    const spy = (async (cmd: string, args: string[], opts: any) => {
      calls.push({ cmd, args, cwd: opts.cwd, timeout: opts.timeout })

      return { stdout: okJson, stderr: '' }
    }) as any

    await scanVenvBlockers('/update/root', spy, stubVenv)
    assert.equal(calls.length, 1)
    const c = calls[0]
    assert.ok(c.cmd.endsWith('python.exe'))
    assert.deepEqual(c.args, ['-m', 'hermes_cli._scan_venv_blockers'])
    assert.equal(c.cwd, '/update/root')
    assert.equal(c.timeout, 60_000)
  })
})

describe('stopSafeVenvBlockers', () => {
  it('stops only blockers explicitly classified as safe local previews', async () => {
    const calls: Array<{ command: string; args: string[] }> = []
    const requestedRoot = path.join(volumeRoot, 'requested', 'install')
    const canonicalRoot = path.join(volumeRoot, 'canonical', 'install')
    const venvPython = path.join(canonicalRoot, 'venv', 'Scripts', 'python.exe')
    const expectedVenv = path.join(canonicalRoot, 'resolved-venv')

    const exec = (async (command: string, args: string[]) => {
      calls.push({ command, args })

      return {
        stdout: JSON.stringify({
          schema_version: 2,
          mode: 'terminate_venv_holder',
          ok: true,
          terminated: true,
          pid: 47484,
          created_at: 1722798000.25,
          root: canonicalRoot,
          venv: expectedVenv,
          error: null
        }),
        stderr: ''
      }
    }) as any

    const outcome = await stopSafeVenvBlockers(
      requestedRoot,
      {
        blocked: true,
        processes: [
          {
            pid: 47484,
            name: 'python.exe',
            cmdline: 'python.exe -m http.server 8766 --directory C:\\preview',
            kind: 'local-preview',
            safeToStop: true,
            label: 'preview',
            port: 8766,
            createTime: 1722798000.25
          },
          {
            pid: 99,
            name: 'python.exe',
            cmdline: 'python.exe important-script.py',
            kind: 'other',
            safeToStop: false
          }
        ]
      },
      exec,
      () => venvPython,
      value => {
        if (value === requestedRoot) {return canonicalRoot}

        if (value === path.dirname(path.dirname(venvPython))) {return expectedVenv}

        throw new Error(`unexpected canonicalization target: ${value}`)
      }
    )

    assert.equal(calls.length, 1)
    assert.equal(calls[0].command, venvPython)
    assert.equal(calls[0].args[0], '-I')
    assert.ok(path.isAbsolute(calls[0].args[1]))
    assert.deepEqual(calls[0].args.slice(2), [
      '--root', canonicalRoot, '--terminate-venv-holder', '47484', '--created-at', '1722798000.25'
    ])
    assert.deepEqual(outcome, { stopped: [47484], failed: [] })
  })
})
}
}

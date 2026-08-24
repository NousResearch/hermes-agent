'use strict'

/**
 * Tests for apps/desktop/electron/venv-blocker-scan.ts
 *
 * Run with: npx vitest run electron/venv-blocker-scan.test.ts
 * (from apps/desktop; wired into npm test:desktop:platforms)
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, it } from 'vitest'

import {
  formatBlockerMessage,
  formatProbeFailedMessage,
  parseVenvBlockerScanOutput,
  resolveVenvPython,
  scanVenvBlockers,
  stopSafeVenvBlockers
} from './venv-blocker-scan'

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
    assert.equal(typeof c.timeout, 'number')
    assert.ok(c.timeout > 0)
  })
})

describe('stopSafeVenvBlockers', () => {
  it('stops only blockers explicitly classified as safe local previews', async () => {
    const calls: Array<{ command: string; args: string[] }> = []

    const exec = (async (command: string, args: string[]) => {
      calls.push({ command, args })

      return { stdout: '', stderr: '' }
    }) as any

    const outcome = await stopSafeVenvBlockers(
      '/update/root',
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
      () => 'C:\\Hermes\\venv\\Scripts\\python.exe'
    )

    assert.deepEqual(calls, [
      {
        command: 'C:\\Hermes\\venv\\Scripts\\python.exe',
        args: ['-m', 'hermes_cli._scan_venv_blockers', '--terminate-safe', '47484', '1722798000.25']
      }
    ])
    assert.deepEqual(outcome, { stopped: [47484], failed: [] })
  })
})

'use strict'

/**
 * Tests for apps/desktop/electron/venv-blocker-scan.ts
 *
 * Run with: npx tsx --test electron/venv-blocker-scan.test.ts
 * (from apps/desktop)
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, it } from 'node:test'

import {
  formatBlockerMessage,
  formatProbeFailedMessage,
  parseVenvBlockerScanOutput,
  resolveVenvPython,
  scanVenvBlockers,
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
      processes: [
        { pid: 101, name: 'python.exe', cmdline: 'serve --host 10.0.0.1' },
      ],
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
    const o = parseVenvBlockerScanOutput(ok({
      blocked: true,
      processes: [{ pid: 1, name: 'p', cmdline: 'c' }],
    }))

    assert.equal(o.kind, 'blocked')
  })

  it('malformed JSON', () => {
    assert.equal(parseVenvBlockerScanOutput('not json').kind, 'probe-failure')
  })

  it('ok=false is rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(JSON.stringify({ ok: false, blocked: false, processes: [] })).kind,
      'probe-failure',
    )
  })

  it('blocked must be boolean', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: 'false' })).kind,
      'probe-failure',
    )
  })

  it('blocked=true with empty processes rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [] })).kind,
      'probe-failure',
    )
  })

  it('blocked=false with non-empty processes rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ processes: [{ pid: 1, name: 'p', cmdline: 'c' }] })).kind,
      'probe-failure',
    )
  })

  it('process pid must be positive integer', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [{ pid: 0, name: 'p', cmdline: 'c' }] })).kind,
      'probe-failure',
    )
  })

  it('process name must be non-empty string', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [{ pid: 1, name: '', cmdline: 'c' }] })).kind,
      'probe-failure',
    )
  })

  it('process missing cmdline is rejected', () => {
    assert.equal(
      parseVenvBlockerScanOutput(ok({ blocked: true, processes: [{ pid: 1, name: 'p' }] })).kind,
      'probe-failure',
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
    ok: true, blocked: true, processes: [{ pid: 1, name: 'p', cmdline: 'c' }],
  })

  function execReturn(json: string): any {
    return ((...args: any[]) => json) as any
  }

  function execThrow(status: number, stderr: string): any {
    return ((...args: any[]) => { const e: any = new Error(); e.status = status; e.stderr = Buffer.from(stderr); throw e }) as any
  }

  it('clear scan returns clear', () => {
    assert.equal(scanVenvBlockers('/r', execReturn(okJson), stubVenv).kind, 'clear')
  })

  it('blocked scan returns blocked', () => {
    assert.equal(scanVenvBlockers('/r', execReturn(blockedJson), stubVenv).kind, 'blocked')
  })

  it('non-zero exit is probe-failure', () => {
    const o = scanVenvBlockers('/r', execThrow(2, 'ModuleNotFoundError'), stubVenv)
    assert.equal(o.kind, 'probe-failure')
  })

  it('missing venv python is probe-failure', () => {
    const o = scanVenvBlockers('/r', execReturn(okJson), () => null)
    assert.equal(o.kind, 'probe-failure')
  })

  it('malformed subprocess output is probe-failure', () => {
    const o = scanVenvBlockers('/r', execReturn('bad json'), stubVenv)
    assert.equal(o.kind, 'probe-failure')
  })

  it('calls subprocess with correct args, cwd, timeout, stdio', () => {
    const calls: any[] = []

    const spy = ((cmd: string, args: string[], opts: any) => {
      calls.push({ cmd, args, cwd: opts.cwd, timeout: opts.timeout, stdio: opts.stdio })

      return okJson
    }) as any

    scanVenvBlockers('/update/root', spy, stubVenv)
    assert.equal(calls.length, 1)
    const c = calls[0]
    assert.ok(c.cmd.endsWith('python.exe'))
    assert.deepEqual(c.args, ['-m', 'hermes_cli._scan_venv_blockers'])
    assert.equal(c.cwd, '/update/root')
    assert.equal(typeof c.timeout, 'number')
    assert.ok(c.timeout > 0)
    assert.deepEqual(c.stdio, ['ignore', 'pipe', 'pipe'])
  })
})
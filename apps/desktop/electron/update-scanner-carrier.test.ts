'use strict'

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, it } from 'vitest'

// The sandbox execution needs a `python` on PATH that can import psutil; the
// scanner refuses to run without it. Skip with a reason instead of failing
// on hosts/runners that only have a bare interpreter.
const pythonHasPsutil = (() => {
  try {
    return spawnSync('python', ['-c', 'import psutil'], { encoding: 'utf-8', windowsHide: true }).status === 0
  } catch {
    return false
  }
})()

import {
  buildUpdateScannerArgv,
  resolveUpdateScannerCarrier,
  UPDATE_SCANNER_RESOURCE_PATH
} from './update-scanner-carrier'
import { parseVenvBlockerScanOutput } from './venv-blocker-scan'

describe('update scanner carrier resolution', () => {
  it('resolves development bytes from candidate source', () => {
    const sourceRoot = path.join(path.parse(process.cwd()).root, 'candidate source')

    assert.equal(
      resolveUpdateScannerCarrier({ sourceRoot, defaultApp: true, resourcesPath: path.join(sourceRoot, 'ignored') }),
      path.resolve(sourceRoot, 'resources', UPDATE_SCANNER_RESOURCE_PATH)
    )
  })

  it('resolves packaged bytes under the stable resources path', () => {
    const resourcesPath = path.join(path.parse(process.cwd()).root, 'Program Files', 'Hermes', 'resources')

    assert.equal(
      resolveUpdateScannerCarrier({ resourcesPath, defaultApp: false }),
      path.resolve(resourcesPath, 'update-scanner', 'scan-venv-blockers.py')
    )
  })

  it('packages the exact candidate scanner at the runtime resource path', () => {
    const desktopRoot = path.resolve(process.cwd())
    const manifest = JSON.parse(fs.readFileSync(path.join(desktopRoot, 'package.json'), 'utf-8'))

    const scannerResource = manifest.build.extraResources.find(
      (resource: { from?: string; to?: string }) => resource.to === UPDATE_SCANNER_RESOURCE_PATH
    )

    assert.deepEqual(scannerResource, {
      from: 'resources/update-scanner/scan-venv-blockers.py',
      to: 'update-scanner/scan-venv-blockers.py'
    })
    assert.equal(fs.existsSync(path.join(desktopRoot, scannerResource.from)), true)
  })

  it.skipIf(!pythonHasPsutil)('materializes and executes isolated argv without importing the target checkout scanner', () => {
    const sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'Hermes carrier [old] & schema-v1 '))

    try {
      const root = path.join(sandbox, 'checkout with spaces & [brackets]')
      const targetPackage = path.join(root, 'hermes_cli')
      fs.mkdirSync(targetPackage, { recursive: true })
      const invokedMarker = path.join(sandbox, 'target-scanner-invoked')
      fs.writeFileSync(
        path.join(targetPackage, '_scan_venv_blockers.py'),
        `from pathlib import Path\nPath(${JSON.stringify(invokedMarker)}).write_text('invoked')\nraise RuntimeError('schema v1 target scanner invoked')\n`
      )

      const pythonProbe = spawnSync('python', ['-c', 'import json,sys; print(json.dumps({"exe":sys.executable,"prefix":sys.prefix}))'], {
        encoding: 'utf-8',
        windowsHide: true
      })

      assert.equal(pythonProbe.status, 0, pythonProbe.stderr)
      const python = JSON.parse(pythonProbe.stdout) as { exe: string; prefix: string }
      fs.symlinkSync(python.prefix, path.join(root, 'venv'), 'junction')

      const carrier = resolveUpdateScannerCarrier({ sourceRoot: path.resolve(process.cwd()), defaultApp: true })
      const argv = buildUpdateScannerArgv(root, carrier)
      assert.deepEqual(argv, ['-I', carrier, '--root', root])

      const child = spawnSync(python.exe, argv, { encoding: 'utf-8', windowsHide: true })
      assert.equal(child.status, 0, child.stderr || child.stdout)
      const output = JSON.parse(child.stdout)

      // The preflight consumes carrier stdout through the exact-envelope parser.
      // A carrier that drifts from hermes_cli/_scan_venv_blockers.py by even one
      // key turns every clean scan into `probe-failure` and aborts the update
      // with "Desktop could not verify the Hermes installation is free".
      const canonicalRoot = fs.realpathSync.native(root)

      const parsed = parseVenvBlockerScanOutput(child.stdout, {
        expectedRoot: canonicalRoot,
        expectedVenv: path.join(canonicalRoot, 'venv')
      })

      assert.notEqual(
        parsed.kind,
        'probe-failure',
        `carrier envelope rejected by Desktop parser: ${(parsed as { error?: string }).error ?? ''}`
      )
      assert.equal(output.schema_version, 2)
      assert.equal(output.mode, 'scan')
      assert.equal(output.ok, true)
      assert.equal(path.resolve(output.root).toLowerCase(), fs.realpathSync.native(root).toLowerCase())
      assert.equal(path.resolve(output.venv).toLowerCase(), path.join(output.root, 'venv').toLowerCase())
      assert.equal(
        fs.realpathSync.native(output.venv).toLowerCase(),
        fs.realpathSync.native(python.prefix).toLowerCase()
      )
      assert.equal(typeof output.blocked, 'boolean')
      assert.ok(Array.isArray(output.processes))
      assert.ok(Array.isArray(output.mcp_bridges))
      assert.equal(fs.existsSync(invokedMarker), false)
    } finally {
      fs.rmSync(sandbox, { recursive: true, force: true })
    }
  })
})

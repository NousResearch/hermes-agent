import assert from 'node:assert/strict'
import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { windowsUpdatePrerequisiteError } from './updater-process'

test('handoff prerequisites fail closed without requiring dependencies, shim or browser UI', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'hermes-prerequisites-'))

  try {
    const python = path.join(root, 'venv', 'Scripts', 'python.exe')
    const script = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')
    assert.match(windowsUpdatePrerequisiteError(root)!, /python.exe/)
    mkdirSync(path.dirname(python), { recursive: true })
    writeFileSync(python, 'file-presence fixture')
    assert.equal(windowsUpdatePrerequisiteError(root), null) // legacy flat layout
    mkdirSync(path.dirname(script), { recursive: true })
    assert.match(windowsUpdatePrerequisiteError(root)!, /windows.ps1/)
    writeFileSync(script, 'file-presence fixture')
    assert.equal(windowsUpdatePrerequisiteError(root), null)
    unlinkSync(python)
    assert.match(windowsUpdatePrerequisiteError(root)!, /python.exe/)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

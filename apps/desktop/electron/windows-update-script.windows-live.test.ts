/**
 * windows-update-script.windows-live.test.ts
 *
 * Live proof for the updater script's binary resolution.
 *
 * scripts/desktop-update/windows.ps1 runs mid-update with the venv's own
 * `Scripts` directory prepended to PATH -- user-writable, and being rewritten
 * by the update in flight. Anything the script starts by bare name is
 * therefore resolved through a directory the update itself is mutating.
 *
 * This does not LAUNCH explorer; it asks PowerShell to resolve the name the
 * way Start-Process would, with a planted decoy first on PATH.
 */

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { describe, it } from 'vitest'

import { windowsPowerShellExecutable } from './windows-powershell-path'

const execFileAsync = promisify(execFile)
const here = path.dirname(fileURLToPath(import.meta.url))
const windowsUpdateScript = path.resolve(here, '..', '..', '..', 'scripts', 'desktop-update', 'windows.ps1')

describe.skipIf(process.platform !== 'win32')('windows.ps1 relaunch binary resolution (live)', () => {
  it('resolves explorer under %SystemRoot%, not through a poisoned PATH', { timeout: 120_000 }, async () => {
    const source = fs.readFileSync(windowsUpdateScript, 'utf8')

    assert.match(
      source,
      /\$explorerPath = Join-Path \$env:SystemRoot 'explorer\.exe'/,
      'the relaunch rung no longer resolves explorer absolutely'
    )
    assert.doesNotMatch(source, /Start-Process -FilePath 'explorer\.exe'/)

    const decoyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-decoy-scripts-'))

    try {
      // Stand-in for `<install>\venv\Scripts`, which the updater prepends to
      // PATH and rewrites while it runs.
      const decoy = path.join(decoyDir, 'explorer.exe')
      fs.copyFileSync(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'where.exe'), decoy)

      const probe = [
        "$ErrorActionPreference = 'Stop'",
        `$env:PATH = '${decoyDir.replace(/'/g, "''")}' + ';' + $env:PATH`,
        "$bare = (Get-Command 'explorer.exe' -ErrorAction SilentlyContinue).Source",
        "$absolute = Join-Path $env:SystemRoot 'explorer.exe'",
        "Write-Output ('BARE=' + $bare)",
        "Write-Output ('ABSOLUTE=' + $absolute)"
      ].join('\n')

      const { stdout } = await execFileAsync(
        windowsPowerShellExecutable(),
        ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', probe],
        { encoding: 'utf8', timeout: 60_000, windowsHide: true }
      )

      const bare = /BARE=(.*)/.exec(String(stdout))?.[1]?.trim() ?? ''
      const absolute = /ABSOLUTE=(.*)/.exec(String(stdout))?.[1]?.trim() ?? ''

      // The bare name resolves to the planted file: that is what the old rung
      // handed to Start-Process.
      assert.equal(bare.toLowerCase(), decoy.toLowerCase())
      assert.notEqual(absolute.toLowerCase(), decoy.toLowerCase())
      assert.equal(fs.existsSync(absolute), true, 'the absolute explorer path must exist')
    } finally {
      fs.rmSync(decoyDir, { recursive: true, force: true })
    }
  })
})

/**
 * update-branch-ref.test.ts
 *
 * The update branch string has two hostile boundaries: git, which reads a
 * leading `-` as an option, and the Windows updater's argv join, where a
 * trailing backslash used to escape its own closing quote. The pure arms pin
 * the validator; the live arm runs the ACTUAL quoting expression from
 * scripts/desktop-update/windows.ps1 through CommandLineToArgvW.
 */

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { describe, it } from 'vitest'

import { isValidUpdateBranchRef, updateBranchRefPattern } from './update-branch-ref'
import { windowsPowerShellExecutable } from './windows-powershell-path'

const execFileAsync = promisify(execFile)
const here = path.dirname(fileURLToPath(import.meta.url))
const windowsUpdateScript = path.resolve(here, '..', '..', '..', 'scripts', 'desktop-update', 'windows.ps1')

describe('update branch validation', () => {
  it('accepts ordinary branch names', () => {
    for (const branch of ['main', 'fork-integration', 'feat/update-marker', 'release-1.2.3', 'user/x_y-z']) {
      assert.equal(isValidUpdateBranchRef(branch), true, branch)
    }

    assert.equal(updateBranchRefPattern('main'), 'refs/heads/main')
  })

  it('rejects a name git would read as an option', () => {
    // `git ls-remote --heads <remote> --upload-pack=...` is the shape this stops.
    assert.equal(isValidUpdateBranchRef('--upload-pack=calc.exe'), false)
    assert.equal(isValidUpdateBranchRef('-main'), false)
  })

  it('rejects a trailing backslash and every other command-line hazard', () => {
    assert.equal(isValidUpdateBranchRef('feature\\'), false)
    assert.equal(isValidUpdateBranchRef('a\\b'), false)
    assert.equal(isValidUpdateBranchRef('a b'), false)
    assert.equal(isValidUpdateBranchRef('a"b'), false)
    assert.equal(isValidUpdateBranchRef('a\nb'), false)
    assert.equal(isValidUpdateBranchRef('a\u0000b'), false)
  })

  it('rejects the git ref-name grammar violations', () => {
    for (const branch of ['a..b', 'a@{b', '@', 'HEAD', '/a', 'a/', 'a//b', 'a.', 'a.lock', '.hidden', 'a~b', 'a^b', 'a:b', 'a?b', 'a*b', 'a[b', '', '  ', ' main']) {
      assert.equal(isValidUpdateBranchRef(branch), false, JSON.stringify(branch))
    }

    assert.equal(isValidUpdateBranchRef(undefined), false)
    assert.equal(isValidUpdateBranchRef(42), false)
  })
})

describe.skipIf(process.platform !== 'win32')('windows.ps1 argv quoting (live)', () => {
  it('round-trips a trailing backslash through CommandLineToArgvW', { timeout: 120_000 }, async () => {
    const source = fs.readFileSync(windowsUpdateScript, 'utf8')
    const match = source.match(/ {4}\$arguments = \(\$HermesArgs \| ForEach-Object \{[\s\S]*?\n {4}\}\) -join ' '/)

    assert.ok(match, 'the argv join expression moved; update this test')

    const probe = `
$ErrorActionPreference = 'Stop'
function Format-HermesArgs([string[]]$HermesArgs) {
${match![0]}
  return $arguments
}
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ArgvProbe {
  [DllImport("shell32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
  private static extern IntPtr CommandLineToArgvW(string cmdLine, out int argc);
  [DllImport("kernel32.dll")] private static extern IntPtr LocalFree(IntPtr h);
  public static string[] Split(string commandLine) {
    int argc;
    IntPtr argv = CommandLineToArgvW(commandLine, out argc);
    if (argv == IntPtr.Zero) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
    try {
      string[] result = new string[argc];
      for (int i = 0; i < argc; i++) result[i] = Marshal.PtrToStringUni(Marshal.ReadIntPtr(argv, i * IntPtr.Size));
      return result;
    } finally { LocalFree(argv); }
  }
}
"@
$cases = @(
  ,@('-m','hermes_cli.main','update','--yes','--branch','feature\\')
  ,@('--branch','C:\\path with space\\')
  ,@('--branch','a"b')
  ,@('--branch','plain/branch')
)
foreach ($case in $cases) {
  $line = '"C:\\exe.exe" ' + (Format-HermesArgs $case)
  $parsed = [ArgvProbe]::Split($line)
  $round = @($parsed[1..($parsed.Length - 1)])
  if (($round -join [char]1) -ne ($case -join [char]1)) {
    Write-Output ('MISMATCH in=[' + ($case -join '|') + '] out=[' + ($round -join '|') + ']')
    exit 1
  }
}
Write-Output 'ROUNDTRIP_OK'
`.trim()

    const { stdout } = await execFileAsync(
      windowsPowerShellExecutable(),
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', probe],
      { encoding: 'utf8', timeout: 90_000, windowsHide: true }
    )

    // The pre-fix expression rendered `feature\` as "feature\", whose closing
    // quote is eaten by the backslash: the branch swallowed the next argument.
    assert.match(String(stdout).trim(), /ROUNDTRIP_OK/)
  })
})

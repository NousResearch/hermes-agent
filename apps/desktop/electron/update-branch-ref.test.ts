/**
 * update-branch-ref.test.ts
 *
 * The update branch string has two hostile boundaries: git, which reads a
 * leading `-` as an option, and the Windows updater's argv join, where a
 * trailing backslash used to escape its own closing quote. The pure arms pin
 * the validator; the live arm runs windows.ps1's production quoting helper
 * through CommandLineToArgvW.
 */

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
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
    for (const branch of [
      'a..b',
      'a@{b',
      '@',
      'HEAD',
      '/a',
      'a/',
      'a//b',
      'a.',
      'a.lock',
      '.hidden',
      'a~b',
      'a^b',
      'a:b',
      'a?b',
      'a*b',
      'a[b',
      '',
      '  ',
      ' main'
    ]) {
      assert.equal(isValidUpdateBranchRef(branch), false, JSON.stringify(branch))
    }

    assert.equal(isValidUpdateBranchRef(undefined), false)
    assert.equal(isValidUpdateBranchRef(42), false)
  })
})

describe.skipIf(process.platform !== 'win32')('windows.ps1 argv quoting (live)', () => {
  it('round-trips a trailing backslash through CommandLineToArgvW', { timeout: 120_000 }, async () => {
    const { stdout } = await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        windowsUpdateScript,
        '-SelfTestArgvQuoting'
      ],
      { encoding: 'utf8', timeout: 90_000, windowsHide: true }
    )

    // The pre-fix expression rendered `feature\` as "feature\", whose closing
    // quote is eaten by the backslash: the branch swallowed the next argument.
    assert.match(String(stdout).trim(), /ARGV-QUOTING SELF-TEST: PASS/)
  })
})

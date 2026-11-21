// Stager warning-line acceptance (mega-PR Target: stager degrade path).
//
// Locks the behavior the fleet verified natively on November 8: when the
// get-windows package lacks its macOS helper binary (`main`), the stager
// refuses the state with the verbatim named warning and leaves nothing
// staged — the build degrades instead of failing. A complete package
// stages whole, helper included.
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test, vi } from 'vitest'
import { stageGetWindowsInto } from './stage-native-deps.mjs'

const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function fixturePackage({ withMain = true } = {}) {
  const base = mkdtempSync(join(tmpdir(), 'stage-get-windows-'))
  roots.push(base)
  const src = join(base, 'pkg')
  mkdirSync(join(src, 'lib'), { recursive: true })
  writeFileSync(join(src, 'package.json'), JSON.stringify({ name: 'get-windows', version: '9.3.0' }))
  writeFileSync(join(src, 'index.js'), 'module.exports = {};\n')
  for (const name of ['linux.js', 'macos.js', 'windows.js']) writeFileSync(join(src, 'lib', name), '// stub\n')
  if (withMain) writeFileSync(join(src, 'main'), '#!/bin/sh\nexit 0\n')
  return { base, src, dest: join(base, 'staged') }
}

test('missing macOS helper warns verbatim and stages nothing (darwin)', () => {
  const { src, dest } = fixturePackage({ withMain: false })
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
  try {
    expect(stageGetWindowsInto(src, dest, { platform: 'darwin', arch: 'arm64' })).toBeUndefined()
    expect(warn).toHaveBeenCalledWith(
      '[stage-native-deps] get-windows is missing its macOS helper binary (main); ' +
        'not staged — read_window_below will be unavailable in this build'
    )
    expect(existsSync(dest)).toBe(false)
  } finally {
    warn.mockRestore()
  }
})

test('complete package stages whole including the helper (darwin)', () => {
  const { src, dest } = fixturePackage({ withMain: true })
  stageGetWindowsInto(src, dest, { platform: 'darwin', arch: 'arm64' })
  expect(existsSync(join(dest, 'main'))).toBe(true)
  expect(existsSync(join(dest, 'index.js'))).toBe(true)
  expect(existsSync(join(dest, 'lib', 'linux.js'))).toBe(true)
})

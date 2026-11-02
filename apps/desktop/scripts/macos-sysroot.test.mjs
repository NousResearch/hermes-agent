import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFileSync: vi.fn(),
}))

const { execFileSync } = await import('node:child_process')
const { macosSysroot, xcrunClangArgv } = await import('./macos-sysroot.mjs')

const exec = vi.mocked(execFileSync)
const opts = env => ({ encoding: 'utf8', env, stdio: 'pipe' })
const XCODE_SDK = 'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
const CLT_SDK = 'SDKs/MacOSX.sdk'

let developerDir, env, warn
const sdkDir = relative => {
  const dir = path.join(developerDir, relative)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'SDKSettings.plist'), '')
  return dir
}
// Answers `xcode-select -p` with the fixture developer dir and `xcrun --sdk X
// --show-sdk-path` with the given result (a string or an Error to throw).
const toolchain = sdkPath => exec.mockImplementation(file => {
  if (file === 'xcode-select') return `${developerDir}\n`
  if (sdkPath instanceof Error) throw sdkPath
  return `${sdkPath}\n`
})

beforeEach(() => {
  developerDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sysroot-'))
  env = { DEVELOPER_DIR: developerDir }
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  exec.mockReset()
  warn.mockRestore()
  fs.rmSync(developerDir, { recursive: true, force: true })
})

it('resolves an explicit SDKROOT path or name through xcrun before any developer default', () => {
  const pinned = sdkDir('SDKs/MacOSX15.4.sdk')
  sdkDir(CLT_SDK)
  for (const SDKROOT of [pinned, 'macosx15.4']) {
    const override = { ...env, SDKROOT }
    toolchain(pinned)
    expect(macosSysroot(override)).toBe(pinned)
    expect(exec).toHaveBeenCalledWith('xcrun', ['--sdk', SDKROOT, '--show-sdk-path'], opts(override))
    expect(exec).not.toHaveBeenCalledWith('xcode-select', expect.anything(), expect.anything())
    expect(xcrunClangArgv(pinned)).toEqual(['clang', '-isysroot', pinned])
  }
  expect(warn).not.toHaveBeenCalled()
})

it('reports a stale SDKROOT and falls back to the toolchain default instead of failing the build', () => {
  const paired = sdkDir(CLT_SDK)
  const missing = path.join(developerDir, 'SDKs/MacOSX99.sdk')
  // xcrun accepts any existing directory as an SDK path; a plain directory is not one.
  const bare = fs.mkdtempSync(path.join(developerDir, 'not-an-sdk-'))
  for (const [SDKROOT, answer] of [
    ['macosx99', new Error('SDK "macosx99" cannot be located')],
    [missing, missing],
    [bare, bare]
  ]) {
    toolchain(answer)
    expect(macosSysroot({ ...env, SDKROOT })).toBe(paired)
    expect(warn).toHaveBeenLastCalledWith(expect.stringContaining(`SDKROOT=${SDKROOT}`))
  }
  expect(warn).toHaveBeenCalledTimes(3)
})

it('ignores an empty SDKROOT', () => {
  const paired = sdkDir(CLT_SDK)
  toolchain(new Error('unexpected xcrun call'))
  expect(macosSysroot({ ...env, SDKROOT: '' })).toBe(paired)
  expect(exec).toHaveBeenCalledTimes(1)
  expect(exec).toHaveBeenLastCalledWith('xcode-select', ['-p'], opts({ ...env, SDKROOT: '' }))
})

it('prefers the Command Line Tools alias over the Xcode platform alias', () => {
  toolchain()
  for (const relative of [XCODE_SDK, CLT_SDK]) {
    const paired = sdkDir(relative)
    expect(macosSysroot(env)).toBe(paired)
    expect(exec).toHaveBeenLastCalledWith('xcode-select', ['-p'], opts(env))
  }
})

it('hands SDK selection back to xcrun when the toolchain has no default alias', () => {
  toolchain()
  expect(macosSysroot(env)).toBeNull()
  // A bare MacOSX.sdk directory with no SDKSettings.plist is not an SDK either.
  fs.mkdirSync(path.join(developerDir, CLT_SDK), { recursive: true })
  expect(macosSysroot(env)).toBeNull()
  expect(xcrunClangArgv(null)).toEqual(['--sdk', 'macosx', 'clang'])
})

it('hands SDK selection back to xcrun when xcode-select fails or answers nothing', () => {
  sdkDir(CLT_SDK)
  exec.mockImplementation(() => { throw new Error('xcode-select: error: unable to get active developer directory') })
  expect(macosSysroot(env)).toBeNull()
  exec.mockReturnValue('\n')
  expect(macosSysroot(env)).toBeNull()
  expect(warn).not.toHaveBeenCalled()
})

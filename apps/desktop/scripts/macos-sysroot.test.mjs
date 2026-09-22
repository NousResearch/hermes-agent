import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFileSync: vi.fn(),
}))

const { execFileSync } = await import('node:child_process')
const { macosSysroot, xcrunClangArgv } = await import('./macos-sysroot.mjs')

afterEach(() => {
  vi.mocked(execFileSync).mockReset()
})

describe('macosSysroot', () => {
  it('prefers an explicit SDKROOT over the developer dir', () => {
    expect(macosSysroot({ SDKROOT: '/pinned/MacOSX.sdk' })).toBe('/pinned/MacOSX.sdk')
    expect(execFileSync).not.toHaveBeenCalled()
  })

  it('names the Command Line Tools SDK paired with the toolchain', () => {
    const developerDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sysroot-clt-'))
    fs.mkdirSync(path.join(developerDir, 'SDKs/MacOSX.sdk'), { recursive: true })
    vi.mocked(execFileSync).mockReturnValue(`${developerDir}\n`)

    expect(macosSysroot({})).toBe(path.join(developerDir, 'SDKs/MacOSX.sdk'))
    fs.rmSync(developerDir, { recursive: true, force: true })
  })

  it('falls back to the Xcode platform SDK layout', () => {
    const developerDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sysroot-xcode-'))
    const sdk = 'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk'
    fs.mkdirSync(path.join(developerDir, sdk), { recursive: true })
    vi.mocked(execFileSync).mockReturnValue(`${developerDir}\n`)

    expect(macosSysroot({})).toBe(path.join(developerDir, sdk))
    fs.rmSync(developerDir, { recursive: true, force: true })
  })

  it('returns null when the developer dir ships no MacOSX.sdk', () => {
    vi.mocked(execFileSync).mockReturnValue(`${os.tmpdir()}/sysroot-no-such-dir\n`)

    expect(macosSysroot({})).toBeNull()
  })
})

describe('xcrunClangArgv', () => {
  it('pins the sysroot when one was resolved', () => {
    expect(xcrunClangArgv('/sdk/MacOSX.sdk')).toEqual(['clang', '-isysroot', '/sdk/MacOSX.sdk'])
  })

  // xcrun options after `clang` would be handed to clang instead of resolving the SDK.
  it('names the SDK before the tool when falling back', () => {
    expect(xcrunClangArgv(null)).toEqual(['--sdk', 'macosx', 'clang'])
  })
})

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFileSync: vi.fn(),
}))

const { execFileSync } = await import('node:child_process')
const { buildCommandScreenshotMonitor } = await import('./build-command-screenshot-monitor.mjs')

afterEach(() => {
  vi.mocked(execFileSync).mockReset()
})

function stageDistDir(prefix) {
  const distDir = fs.mkdtempSync(path.join(os.tmpdir(), prefix))
  const staging = path.resolve(distDir, `native/command-screenshot-monitor.${process.pid}.tmp`)
  fs.mkdirSync(path.dirname(staging), { recursive: true })
  fs.writeFileSync(staging, 'staged')
  return distDir
}

// The helper shells out to xcrun; pin the argv contract (not the toolchain),
// so a non-macOS CI host still proves what the macOS build will run.
describe('buildCommandScreenshotMonitor argv', () => {
  it('links against the sysroot paired with the active toolchain', () => {
    const distDir = stageDistDir('csm-argv-')

    const out = buildCommandScreenshotMonitor({
      distDir,
      platform: 'darwin',
      sysroot: '/Developer/SDKs/MacOSX.sdk',
    })

    expect(execFileSync).toHaveBeenCalledOnce()
    const [cmd, argv] = vi.mocked(execFileSync).mock.calls[0]
    expect(cmd).toBe('xcrun')
    expect(argv.slice(0, 3)).toEqual(['clang', '-isysroot', '/Developer/SDKs/MacOSX.sdk'])
    expect(out).toBe(path.resolve(distDir, 'native/command-screenshot-monitor'))
    fs.rmSync(distDir, { recursive: true, force: true })
  })

  it('falls back to the default SDK when no paired sysroot exists', () => {
    const distDir = stageDistDir('csm-fallback-')

    buildCommandScreenshotMonitor({ distDir, platform: 'darwin', sysroot: null })

    const [, argv] = vi.mocked(execFileSync).mock.calls[0]
    // `--sdk macosx` must precede the tool name: xcrun options after `clang`
    // would be handed to clang instead of resolving the SDK.
    expect(argv.slice(0, 3)).toEqual(['--sdk', 'macosx', 'clang'])
    fs.rmSync(distDir, { recursive: true, force: true })
  })

  it('is a no-op off macOS', () => {
    const distDir = path.join(os.tmpdir(), 'csm-never')

    expect(buildCommandScreenshotMonitor({ distDir, platform: 'linux' })).toBeNull()
    expect(execFileSync).not.toHaveBeenCalled()
  })
})

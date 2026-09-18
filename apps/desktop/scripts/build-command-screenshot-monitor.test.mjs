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

// The helper shells out to xcrun; pin the argv contract (not the toolchain),
// so a non-macOS CI host still proves what the macOS build will run.
describe('buildCommandScreenshotMonitor argv', () => {
  it('names the macOS SDK explicitly so driver and linker agree', () => {
    const distDir = fs.mkdtempSync(path.join(os.tmpdir(), 'csm-argv-'))
    const staging = path.resolve(distDir, `native/command-screenshot-monitor.${process.pid}.tmp`)
    fs.mkdirSync(path.dirname(staging), { recursive: true })
    fs.writeFileSync(staging, 'staged')

    const out = buildCommandScreenshotMonitor({ distDir, platform: 'darwin' })

    expect(execFileSync).toHaveBeenCalledOnce()
    const [cmd, argv] = vi.mocked(execFileSync).mock.calls[0]
    expect(cmd).toBe('xcrun')
    // `--sdk macosx` must precede the tool name: xcrun options after `clang`
    // would be handed to clang instead of resolving the SDK.
    expect(argv.slice(0, 3)).toEqual(['--sdk', 'macosx', 'clang'])
    expect(out).toBe(path.resolve(distDir, 'native/command-screenshot-monitor'))
    fs.rmSync(distDir, { recursive: true, force: true })
  })

  it('is a no-op off macOS', () => {
    const distDir = path.join(os.tmpdir(), 'csm-never')

    expect(buildCommandScreenshotMonitor({ distDir, platform: 'linux' })).toBeNull()
    expect(execFileSync).not.toHaveBeenCalled()
  })
})

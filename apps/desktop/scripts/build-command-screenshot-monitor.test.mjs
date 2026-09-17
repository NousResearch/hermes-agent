import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it, vi } from 'vitest'

vi.mock('node:child_process', () => ({
  execFileSync: vi.fn(),
}))

const { execFileSync } = await import('node:child_process')
const { buildCommandScreenshotMonitor } = await import('./build-command-screenshot-monitor.mjs')

// The helper shells out to xcrun; pin the argv contract (not the toolchain).
describe('buildCommandScreenshotMonitor argv', () => {
  it('names the macOS SDK explicitly so driver and linker agree (#113708)', () => {
    if (process.platform !== 'darwin') {
      return
    }
    const distDir = fs.mkdtempSync(path.join(os.tmpdir(), 'csm-argv-'))
    const staging = path.resolve(distDir, `native/command-screenshot-monitor.${process.pid}.tmp`)
    fs.mkdirSync(path.dirname(staging), { recursive: true })
    fs.writeFileSync(staging, 'staged')

    const out = buildCommandScreenshotMonitor({ distDir })

    expect(execFileSync).toHaveBeenCalledOnce()
    const [cmd, argv] = execFileSync.mock.calls[0]
    expect(cmd).toBe('xcrun')
    expect(argv.slice(0, 3)).toEqual(['--sdk', 'macosx', 'clang'])
    expect(out).toBe(path.resolve(distDir, 'native/command-screenshot-monitor'))
    fs.rmSync(distDir, { recursive: true, force: true })
  })
})

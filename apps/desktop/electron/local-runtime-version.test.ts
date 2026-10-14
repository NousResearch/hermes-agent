import { describe, expect, it, vi } from 'vitest'

import { PROBE_TIMEOUT_MS } from './backend-probes'
import { parseLocalRuntimeVersion, resolveLocalRuntimeVersion, type VersionProbeRunner } from './local-runtime-version'

describe('local client version source', (): void => {
  it('parses the exact version line emitted by the local launcher', (): void => {
    expect(parseLocalRuntimeVersion('Hermes Agent v0.21.5+6644.gb412ee9 (2026.9.24)\n')).toBe('0.21.5+6644.gb412ee9')
    expect(parseLocalRuntimeVersion('gateway: v0.21.5+6634\n')).toBe('')
  })

  it('probes the installation launcher with a bounded command and ignores remote values', async (): Promise<void> => {
    const run: VersionProbeRunner = vi.fn(async (_command, args, options) => {
      expect(args).toEqual(['--version'])
      expect(options.cwd).toBe('/isolated/hermes')
      expect(options.timeout).toBe(PROBE_TIMEOUT_MS)
      expect(options.maxBuffer).toBe(64 * 1024)
      expect(options.env.HERMES_HOME).toBe('/isolated/home')
      return 'Hermes Agent v0.21.5+6644.gb412ee9 (local)\n'
    })

    await expect(resolveLocalRuntimeVersion('/isolated/hermes', '/isolated/home', {
      launcher: '/isolated/hermes/.hermes/bin/hermes',
      run
    })).resolves.toBe('0.21.5+6644.gb412ee9')
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('fails closed when the local launcher cannot answer', async (): Promise<void> => {
    const run: VersionProbeRunner = vi.fn(async () => {
      throw new Error('timeout')
    })

    await expect(resolveLocalRuntimeVersion('/isolated/hermes', '/isolated/home', {
      launcher: '/isolated/hermes/.hermes/bin/hermes',
      run
    })).resolves.toBe('')
  })

  it('does not invoke a missing installation launcher', async (): Promise<void> => {
    const run: VersionProbeRunner = vi.fn()

    await expect(resolveLocalRuntimeVersion('/isolated/hermes', '/isolated/home', { launcher: null, run })).resolves.toBe('')
    expect(run).not.toHaveBeenCalled()
  })

})

import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, it, vi } from 'vitest'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import { readSourceUpdate, type SourceUpdate } from './checkout-source'

it('offers manual recovery only for a missing source probe, never for a broken probe', async (): Promise<void> => {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'legacy-channel-'))
  const home: string = path.join(root, 'profile')
  const modulePath: string = path.join(root, 'hermes_cli', 'source_releases.py')
  fs.mkdirSync(path.dirname(modulePath))
  fs.mkdirSync(home)
  fs.writeFileSync(path.join(root, 'hermes_cli', '__init__.py'), '')

  const probe: () => Promise<SourceUpdate | null> = (): Promise<SourceUpdate | null> => readSourceUpdate({
    python: process.env.HERMES_PYTHON || 'python3', git: 'git', updateRoot: root, hermesHome: home
  })

  const deps: CheckoutStrategyDeps = {
    isGitCheckout: (): boolean => true,
    updateCheckCachePath: path.join(root, 'cache.json'),
    writeFileAtomic: vi.fn(), readSourceUpdate: probe, fetchGitHubApi: vi.fn(),
    hermesHome: home, isWindows: process.platform === 'win32', isMac: process.platform === 'darwin',
    defaultUpdateBranch: 'main', updateHandoffDwellMs: 0,
    directoryExists: fs.existsSync, readCanonicalInstallStamp: (): null => null,
    readDesktopUpdateConfig: (): { branch: string } => ({ branch: 'main' }),
    resolveUpdateRoot: (): string => root, resolveUpdaterBinary: vi.fn((): string => 'frozen-updater'),
    resolveHealedBranch: vi.fn(async (_root: string, branch: string): Promise<string> => branch),
    getOriginUrl: async (): Promise<string> => 'https://github.com/fixture/repo',
    runGit: async (args: string[]): Promise<{ code: number; stdout: string; stderr: string }> => ({
      code: 0, stdout: args.includes('--abbrev-ref') ? 'feature/work' : args.includes('HEAD') ? 'a'.repeat(40) : '', stderr: ''
    }),
    firstLine: (text: string): string => text.split('\n')[0], pathWithVenvBin: (): string => '',
    venvHermesShimPath: (): string => '', emitUpdateProgress: vi.fn(), rememberLog: vi.fn(),
    startHermes: vi.fn(async (): Promise<void> => {}), startGatewaysAfterUpdateAbort: vi.fn(),
    releaseBackendLockForUpdate: vi.fn(async (): Promise<{ unlocked: boolean }> => ({ unlocked: true })),
    repairMacUpdaterHelper: vi.fn(), preflightStateDb: vi.fn(), runningAppBundle: (): null => null,
    markQuittingForHandoff: vi.fn(), quit: vi.fn()
  }

  const strategy: ReturnType<typeof createCheckoutStrategy> = createCheckoutStrategy(deps)

  try {
    for (const oldModule of ['def resolve_source_release(channel):\n    return None, None\n', null]) {
      if (oldModule !== null) { fs.writeFileSync(modulePath, oldModule) }
      else { fs.rmSync(modulePath) }

      expect(await strategy.check()).toMatchObject({ supported: false, reason: 'source-probe-unavailable' })
      const result: Awaited<ReturnType<typeof strategy.apply>> = await strategy.apply({})
      expect(result).toMatchObject({ manual: true, command: 'hermes update --help' })
      expect(result.message).toContain('branch or channel')
      expect(result.command).not.toContain('--branch')
      expect(deps.releaseBackendLockForUpdate).not.toHaveBeenCalled()
      expect(deps.resolveUpdaterBinary).not.toHaveBeenCalled()
      expect(deps.fetchGitHubApi).not.toHaveBeenCalled()
      expect(deps.quit).not.toHaveBeenCalled()
    }

    fs.writeFileSync(modulePath, 'def main():\n    raise RuntimeError("invalid channel configuration")\n')
    await expect(strategy.apply({})).rejects.toThrow('invalid channel configuration')
    fs.writeFileSync(modulePath, 'import missing_probe_dependency\n')
    await expect(probe()).rejects.toThrow('missing_probe_dependency')
    expect(deps.releaseBackendLockForUpdate).not.toHaveBeenCalled()
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

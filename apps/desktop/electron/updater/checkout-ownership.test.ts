import { describe, expect, it, vi } from 'vitest'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'

function dependencies(): CheckoutStrategyDeps {
  return {
    isGitCheckout: (): boolean => true,
    updateCheckCachePath: 'unused-cache.json',
    writeFileAtomic: vi.fn(),
    readSourceUpdate: vi.fn(async (): Promise<{ channel: 'main' }> => ({ channel: 'main' })),
    fetchGitHubApi: vi.fn(),
    hermesHome: 'home',
    isWindows: process.platform === 'win32',
    isMac: process.platform === 'darwin',
    defaultUpdateBranch: 'main',
    updateHandoffDwellMs: 0,
    directoryExists: () => true,
    readCanonicalInstallStamp: () => ({ updateMechanism: 'external' }),
    readDesktopUpdateConfig: () => ({ branch: 'main' }),
    resolveUpdateRoot: () => 'repo',
    resolveUpdaterBinary: () => null,
    resolveHealedBranch: async (_, branch) => branch,
    getOriginUrl: async () => '',
    runGit: vi.fn(async () => { throw new Error('unexpected git invocation') }),
    firstLine: text => text.split('\n')[0],

    pathWithVenvBin: () => '',
    venvHermesShimPath: () => '',
    emitUpdateProgress: vi.fn(),
    rememberLog: vi.fn(),
    startHermes: vi.fn(async () => {}),
    startGatewaysAfterUpdateAbort: vi.fn(),
    releaseBackendLockForUpdate: vi.fn(async () => ({ unlocked: true })),
    repairMacUpdaterHelper: vi.fn(),
    preflightStateDb: vi.fn(),
    runningAppBundle: () => null,
    markQuittingForHandoff: vi.fn(),
    quit: vi.fn()
  }
}

describe('checkout update admission', () => {
  it.each(['external', 'app-installer', 'electron-updater'] as const)('refuses %s-owned code without fetching or stopping the backend', async updateMechanism => {
    const deps = dependencies()
    deps.readCanonicalInstallStamp = () => ({ updateMechanism })
    const strategy = createCheckoutStrategy(deps)
    const result = await strategy.check()

    expect(result.supported).toBe(false)
    expect(await strategy.apply({})).toMatchObject({ ok: false })
    expect(deps.readSourceUpdate).not.toHaveBeenCalled()
    expect(result.mechanism).toBe(strategy.mechanism)
    expect(deps.runGit).not.toHaveBeenCalled()
    expect(deps.releaseBackendLockForUpdate).not.toHaveBeenCalled()
    expect(deps.quit).not.toHaveBeenCalled()
  })

  it('rejects a missing source checkout without attempting git', async () => {
    const deps = dependencies()
    deps.isGitCheckout = (): boolean => false
    const result = await createCheckoutStrategy(deps).check()

    expect(result.reason).toBe('not-a-git-checkout')
    expect(deps.runGit).not.toHaveBeenCalled()
  })
})

import { expect, it, vi } from 'vitest'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import type { SourceUpdate } from './checkout-source'

it.each(['not-a-git-checkout', 'update-root-steward-owned-git-tree', 'fetch-failed'])(
  'preserves the Python refusal or error before handoff: %s',
  async (reason: string): Promise<void> => {
    const status: SourceUpdate =
      reason === 'fetch-failed' ? { supported: true, error: reason } : { supported: false, reason }

    const deps: CheckoutStrategyDeps = {
      readSourceUpdate: vi.fn(async (): Promise<SourceUpdate> => status),
      hermesHome: 'home',
      isWindows: process.platform === 'win32',
      isMac: process.platform === 'darwin',
      defaultUpdateBranch: 'main',
      updateHandoffDwellMs: 0,
      resolveUpdateRoot: (): string => 'repo',
      resolveUpdaterBinary: vi.fn((): null => null),
      remoteGatewayActive: (): boolean => false,
      emitUpdateProgress: vi.fn(),
      rememberLog: vi.fn(),
      startHermes: vi.fn(async (): Promise<void> => {}),
      stopBackendsForUpdate: vi.fn(async (): Promise<void> => {}),
      repairMacUpdaterHelper: vi.fn(),
      preflightStateDb: vi.fn(),
      runningAppBundle: (): null => null,
      markQuittingForHandoff: vi.fn(),
      quit: vi.fn()
    }

    const strategy: ReturnType<typeof createCheckoutStrategy> = createCheckoutStrategy(deps)
    expect(await strategy.check()).toMatchObject({ ...status, mechanism: strategy.mechanism })
    expect(await strategy.apply()).toMatchObject({ ok: false, error: reason })
    expect(deps.readSourceUpdate).toHaveBeenLastCalledWith('repo', { force: true })
    expect(deps.resolveUpdaterBinary).not.toHaveBeenCalled()
    expect(deps.stopBackendsForUpdate).not.toHaveBeenCalled()
    expect(deps.quit).not.toHaveBeenCalled()
  }
)

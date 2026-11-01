import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, it, vi } from 'vitest'

import { type CheckoutStrategyDeps, createCheckoutStrategy, readStampedCommit } from './checkout'
import type { SourceUpdate } from './checkout-source'

const STAMPED: string = 'a'.repeat(40)
const PROBED: string = 'b'.repeat(40)

function writeStamp(root: string, commit: string): void {
  fs.writeFileSync(path.join(root, 'install-stamp.json'), JSON.stringify({ schemaVersion: 2, commit, source: 'git' }))
}

function deps(root: string, status: SourceUpdate | null, isWindows: boolean): CheckoutStrategyDeps {
  return {
    readSourceUpdate: vi.fn(async (): Promise<SourceUpdate | null> => status),
    hermesHome: 'home',
    isWindows,
    isMac: process.platform === 'darwin',
    defaultUpdateBranch: 'main',
    updateHandoffDwellMs: 0,
    resolveUpdateRoot: (): string => root,
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
}

it.each([
  ['windows-handoff', true],
  ['posix-handoff', false]
])(
  '%s reports the checkout commit even when the source probe never supplied one',
  async (_mechanism: string, isWindows: boolean): Promise<void> => {
    const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-sha-'))

    try {
      // A probe-less checkout (the manual path: no hermes_cli/source_check.py)
      // previously reported no currentSha at all — the commit showed nowhere.
      writeStamp(root, STAMPED)
      const strategy: ReturnType<typeof createCheckoutStrategy> = createCheckoutStrategy(deps(root, null, isWindows))

      expect(await strategy.check()).toMatchObject({
        supported: true,
        mechanism: isWindows ? 'windows-handoff' : 'posix-handoff',
        currentSha: STAMPED
      })
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }
)

it('fills currentSha from the install stamp when the probe omitted it, but never overrides the probe', async (): Promise<void> => {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-sha-'))

  try {
    writeStamp(root, STAMPED)

    // A probe status that carries no currentSha (stale cache shapes, error
    // statuses) still leaves the badge with a commit to show.
    const bare: SourceUpdate = { supported: true, error: 'fetch-failed', behind: null }
    expect(await createCheckoutStrategy(deps(root, bare, false)).check()).toMatchObject({ currentSha: STAMPED })

    // The probe's own answer — including its currentSha — always wins.
    const probed: SourceUpdate = { supported: true, currentSha: PROBED, behind: 0, updateAvailable: false }
    expect(await createCheckoutStrategy(deps(root, probed, false)).check()).toMatchObject({ currentSha: PROBED })
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

it('readStampedCommit returns the stamped commit, or null for absent and malformed stamps', (): void => {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-stamp-'))

  try {
    expect(readStampedCommit(root)).toBeNull()

    writeStamp(root, STAMPED)
    expect(readStampedCommit(root)).toBe(STAMPED)

    fs.writeFileSync(path.join(root, 'install-stamp.json'), JSON.stringify({ commit: null, source: 'git' }))
    expect(readStampedCommit(root)).toBeNull()

    fs.writeFileSync(path.join(root, 'install-stamp.json'), '{not json')
    expect(readStampedCommit(root)).toBeNull()
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

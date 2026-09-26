import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopAutoUpdateClaim, DesktopUpdateApplyResult, DesktopUpdateStatus } from '@/global'

const updates = vi.hoisted(() => ({
  checkUpdates: vi.fn(),
  applyUpdates: vi.fn()
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

vi.mock('@/store/updates', async () => {
  const { atom } = await import('nanostores')

  return {
    $updateApply: atom({ applying: false }),
    $updateChecking: atom(false),
    $updateOverlayOpen: atom(false),
    $updateOverlayTarget: atom('client'),
    $updateStatus: atom(null),
    checkUpdates: updates.checkUpdates,
    applyUpdates: updates.applyUpdates
  }
})

import { notify } from '@/store/notifications'
import { $updateChecking, $updateOverlayOpen, $updateStatus } from '@/store/updates'

import { _resetAutoUpdateForTests, autoUpdateCheckVerdict, runAutoUpdateOnLaunch } from './auto-update'

const behind = (extra: Partial<DesktopUpdateStatus> = {}): DesktopUpdateStatus => ({
  supported: true,
  behind: 3,
  targetSha: 'abc',
  branch: 'main',
  ...extra
})

describe('autoUpdateCheckVerdict', () => {
  it('applies when the selected channel has something newer', () => {
    expect(autoUpdateCheckVerdict(behind())).toBe('apply')
    expect(autoUpdateCheckVerdict({ supported: true, updateAvailable: true, latestTag: 'v0.22.0' })).toBe('apply')
  })

  it('reports up-to-date instead of applying', () => {
    expect(autoUpdateCheckVerdict(behind({ behind: 0 }))).toBe('up-to-date')
  })

  it('never auto-updates a checkout with local edits', () => {
    expect(autoUpdateCheckVerdict(behind({ dirty: true }))).toBe('skipped-dirty')
  })

  it('treats errors, unsupported installs, and retired builds as terminal for the session', () => {
    expect(autoUpdateCheckVerdict(null)).toBe('check-failed')
    expect(autoUpdateCheckVerdict(behind({ error: 'check-failed' }))).toBe('check-failed')
    expect(autoUpdateCheckVerdict(behind({ supported: false }))).toBe('skipped-unsupported')
    expect(
      autoUpdateCheckVerdict(behind({ retirement: { state: 'discontinued', destination: 'x', version: '1' } }))
    ).toBe('skipped-unsupported')
  })
})

describe('runAutoUpdateOnLaunch', () => {
  let auto: {
    get: ReturnType<typeof vi.fn>
    set: ReturnType<typeof vi.fn>
    claim: ReturnType<typeof vi.fn<() => Promise<DesktopAutoUpdateClaim>>>
    report: ReturnType<typeof vi.fn>
  }

  beforeEach(() => {
    _resetAutoUpdateForTests()
    vi.clearAllMocks()
    $updateOverlayOpen.set(false)
    $updateStatus.set(null)
    $updateChecking.set(false)
    auto = {
      get: vi.fn(),
      set: vi.fn(),
      claim: vi.fn<() => Promise<DesktopAutoUpdateClaim>>(),
      report: vi.fn(async () => ({ enabled: true, supported: true, sessionScope: 'login', lastAttempt: null }))
    }
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { updates: { auto } }
  })

  afterEach(() => {
    vi.useRealTimers()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  const settle = () => vi.waitFor(() => expect(auto.report).toHaveBeenCalled())

  it('runs the ordinary check + apply flow and reports the hand-off', async () => {
    auto.claim.mockResolvedValue({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'S1' })
    updates.checkUpdates.mockResolvedValue(behind())
    updates.applyUpdates.mockResolvedValue({ ok: true, handedOff: true } satisfies DesktopUpdateApplyResult)

    runAutoUpdateOnLaunch()
    await settle()

    expect(updates.checkUpdates).toHaveBeenCalledWith({ force: true })
    expect(updates.applyUpdates).toHaveBeenCalledTimes(1)
    expect($updateOverlayOpen.get()).toBe(true)
    expect(auto.report).toHaveBeenCalledWith({ sessionKey: 'S1', outcome: 'handed-off', target: 'main' })
  })

  it('does not apply when already current', async () => {
    auto.claim.mockResolvedValue({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'S1' })
    updates.checkUpdates.mockResolvedValue(behind({ behind: 0 }))

    runAutoUpdateOnLaunch()
    await settle()

    expect(updates.applyUpdates).not.toHaveBeenCalled()
    expect(auto.report).toHaveBeenCalledWith(expect.objectContaining({ outcome: 'up-to-date' }))
  })

  it('reports a failed apply once (main keeps it from retrying this session)', async () => {
    auto.claim.mockResolvedValue({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'S1' })
    updates.checkUpdates.mockResolvedValue(behind())
    updates.applyUpdates.mockResolvedValue({ ok: false, error: 'apply-failed', message: 'network' })

    runAutoUpdateOnLaunch()
    await settle()

    expect(auto.report).toHaveBeenCalledWith(
      expect.objectContaining({ sessionKey: 'S1', outcome: 'failed', message: 'network' })
    )
  })

  it('does nothing when main says skip', async () => {
    auto.claim.mockResolvedValue({ action: 'skip', reason: 'already-ran-this-session', sessionKey: 'S1' })

    runAutoUpdateOnLaunch()
    await vi.waitFor(() => expect(auto.claim).toHaveBeenCalled())
    await Promise.resolve()

    expect(updates.checkUpdates).not.toHaveBeenCalled()
    expect(auto.report).not.toHaveBeenCalled()
  })

  it('waits behind a busy gateway, then runs once it drains', async () => {
    vi.useFakeTimers()
    auto.claim
      .mockResolvedValueOnce({ action: 'defer', reason: 'busy', sessionKey: 'S1', retryInMs: 60_000, activeAgents: 1 })
      .mockResolvedValueOnce({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'S1' })
    updates.checkUpdates.mockResolvedValue(behind({ behind: 0 }))

    runAutoUpdateOnLaunch()
    await vi.waitFor(() => expect(notify).toHaveBeenCalledTimes(1))
    expect(updates.checkUpdates).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(60_000)
    await vi.waitFor(() => expect(auto.report).toHaveBeenCalled())

    expect(auto.claim).toHaveBeenCalledTimes(2)
  })

  it('reuses a check that just answered instead of hitting the server twice at launch', async () => {
    auto.claim.mockResolvedValue({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'S1' })
    // The mount-time passive check (or the user's own Check now) already answered.
    $updateStatus.set(behind({ fetchedAt: Date.now() - 5_000 }))
    updates.applyUpdates.mockResolvedValue({ ok: true, handedOff: true } satisfies DesktopUpdateApplyResult)

    runAutoUpdateOnLaunch()
    await settle()

    expect(updates.checkUpdates).not.toHaveBeenCalled()
    expect(updates.applyUpdates).toHaveBeenCalledTimes(1)
  })

  it('waits for an in-flight check rather than racing it', async () => {
    auto.claim.mockResolvedValue({ action: 'run', reason: 'first-launch-after-login', sessionKey: 'S1' })
    $updateChecking.set(true)
    updates.checkUpdates.mockResolvedValue(behind({ behind: 0 }))

    runAutoUpdateOnLaunch()
    await vi.waitFor(() => expect(auto.claim).toHaveBeenCalled())
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(updates.checkUpdates).not.toHaveBeenCalled()

    // The manual check finishes with a stale answer → the auto check forces its own.
    $updateStatus.set(behind({ behind: 0, fetchedAt: Date.now() - 10 * 60_000 }))
    $updateChecking.set(false)
    await settle()

    expect(updates.checkUpdates).toHaveBeenCalledWith({ force: true })
    expect(updates.applyUpdates).not.toHaveBeenCalled()
  })

  it('never retries a deferral faster than once a minute, whatever main says', async () => {
    vi.useFakeTimers()
    auto.claim
      .mockResolvedValueOnce({ action: 'defer', reason: 'busy', sessionKey: 'S1', retryInMs: 0, activeAgents: 1 })
      .mockResolvedValue({ action: 'skip', reason: 'already-ran-this-session', sessionKey: 'S1' })

    runAutoUpdateOnLaunch()
    await vi.waitFor(() => expect(auto.claim).toHaveBeenCalledTimes(1))

    await vi.advanceTimersByTimeAsync(59_000)
    expect(auto.claim).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1_000)
    await vi.waitFor(() => expect(auto.claim).toHaveBeenCalledTimes(2))
  })

  it('only starts once per renderer', async () => {
    auto.claim.mockResolvedValue({ action: 'skip', reason: 'disabled', sessionKey: '' })

    runAutoUpdateOnLaunch()
    runAutoUpdateOnLaunch()
    await vi.waitFor(() => expect(auto.claim).toHaveBeenCalled())

    expect(auto.claim).toHaveBeenCalledTimes(1)
  })
})

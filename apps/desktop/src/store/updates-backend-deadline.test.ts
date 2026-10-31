import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The remote backend update is bounded by the absence of progress (#137848):
// a receipt that keeps advancing through stages keeps the apply alive even
// when the log is quiet, and when a bound does expire the receipt and the
// backend's version give the verdict, never the clock alone.

vi.mock('@/lib/storage', () => ({
  persistBoolean: () => undefined,
  persistString: () => undefined,
  readJson: () => null,
  writeJson: () => undefined,
  storedBoolean: (_key: string, fallback: boolean) => fallback,
  storedString: () => null
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), dismissNotification: vi.fn() }))

const { atom } = await import('nanostores')

vi.mock('@/store/connections', () => ({
  $connectionsRegistry: atom<unknown>(null),
  refreshConnectionsRegistry: () => Promise.resolve(null)
}))

const checkHermesUpdateSpy = vi.fn()
const updateHermesSpy = vi.fn()
const getActionStatusSpy = vi.fn()

vi.mock('@/hermes', () => ({
  checkHermesUpdate: (...args: unknown[]) => checkHermesUpdateSpy(...args),
  updateHermes: (...args: unknown[]) => updateHermesSpy(...args),
  getActionStatus: (...args: unknown[]) => getActionStatusSpy(...args)
}))

vi.mock('@/store/gateway-reconnect', () => ({ reconnectGateway: vi.fn().mockResolvedValue(undefined) }))

const { applyBackendUpdate, $backendUpdateApply, $backendUpdateStatus } = await import('./updates')
const { BUSY_STAGE_STALL_MS } = await import('./update-stage-deadlines')

const ACTION = 'e'.repeat(32)
const MINUTE = 60_000

function running(stage: string | null, stageAt: string | null) {
  return {
    exit_code: null,
    lines: ['=== hermes-update started now ===', 'Fetching updates...'],
    name: 'hermes-update',
    pid: 1,
    running: true,
    receipt: {
      outcome: 'running',
      started_at: new Date().toISOString(),
      finished_at: null,
      action_id: ACTION,
      pre_sha: null,
      post_sha: null,
      post_version: null,
      fleet_states: [],
      stage,
      stage_at: stageAt
    }
  }
}

describe('remote backend update deadline', () => {
  beforeEach(() => {
    checkHermesUpdateSpy.mockReset()
    updateHermesSpy.mockReset().mockResolvedValue({ action_id: ACTION, ok: true, name: 'hermes-update', pid: 1 })
    getActionStatusSpy.mockReset()
    $backendUpdateStatus.set(null)
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('a log-silent update whose receipt keeps advancing stages is never failed', async () => {
    // 30 minutes of an unchanged log tail: only the receipt's stage boundaries move.
    let polls = 0

    getActionStatusSpy.mockImplementation(async () => {
      polls += 1
      const stage = Math.floor(polls / 200)

      return running(`stage-${stage}`, `t${stage}`)
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(30 * MINUTE)

    expect($backendUpdateApply.get().applying).toBe(true)
    expect($backendUpdateApply.get().stage).toBe('pull')

    getActionStatusSpy.mockImplementation(async () => ({
      ...running('completion', 'tz'),
      running: false,
      receipt: { ...running('completion', 'tz').receipt, outcome: 'success', finished_at: new Date().toISOString() }
    }))
    await vi.advanceTimersByTimeAsync(5000)
    await expect(promise).resolves.toMatchObject({ ok: true })
  })

  it('output under a fixed last line still counts once the log is at its cap', async () => {
    // A gateway drain at the 2,000-line cap: a countdown line changes every 30 s
    // above an unchanged last line, and the receipt stage does not move.
    getActionStatusSpy.mockImplementation(async () => {
      const countdown = Math.floor(Date.now() / 30_000)
      const lines = Array.from({ length: 2000 }, (_line, index) => `line ${countdown + index}`)

      return {
        ...running('drain', 't0'),
        lines: [...lines.slice(1, -1), `draining: ${countdown}`, 'Waiting for gateways...']
      }
    })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(3 * BUSY_STAGE_STALL_MS.pull)

    expect($backendUpdateApply.get().applying).toBe(true)
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()

    getActionStatusSpy.mockImplementation(async () => ({
      ...running('completion', 'tz'),
      running: false,
      receipt: { ...running('completion', 'tz').receipt, outcome: 'success', finished_at: new Date().toISOString() }
    }))
    await vi.advanceTimersByTimeAsync(5000)
    await expect(promise).resolves.toMatchObject({ ok: true })
  })

  it('a stalled action whose backend already reached the target is a success, not a failure', async () => {
    getActionStatusSpy.mockResolvedValue(running('fetch', 't0'))
    checkHermesUpdateSpy.mockResolvedValue({ behind: 0, current_version: '9.9.9', commits: [] })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(BUSY_STAGE_STALL_MS.pull + 5000)

    await expect(promise).resolves.toMatchObject({ ok: true })
  })

  it('a stalled action with no proof of the update fails as stalled within the inactivity bound', async () => {
    getActionStatusSpy.mockResolvedValue(running('fetch', 't0'))
    checkHermesUpdateSpy.mockResolvedValue({ behind: 3, current_version: '1.0.0', commits: [{ sha: 'a'.repeat(40) }] })

    const promise = applyBackendUpdate()
    await vi.advanceTimersByTimeAsync(BUSY_STAGE_STALL_MS.pull + 5000)

    await expect(promise).resolves.toMatchObject({ ok: false, error: 'apply-failed' })
    expect($backendUpdateApply.get().stage).toBe('error')
    expect($backendUpdateApply.get().message).toMatch(/stopped reporting progress/)
  })
})

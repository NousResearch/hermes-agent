import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionResumeResult } from '@/types/hermes'

import {
  beginReconnectResumeEpoch,
  ensureLiveSessionsResumed,
  type LiveResumeSurface,
  RECONNECT_RESUME_SETTLING_RETRY_MS,
  reconnectFollowUpRuntimeId,
  type ReconnectResumePorts,
  resetReconnectResumeForTests
} from './reconnect-resume'
import { $sessionStates } from './session-states'

const gatewayError = (code: number, reason?: string) =>
  Object.assign(new Error(`gateway refused ${code}`), reason ? { code, data: { reason } } : { code })

const resumeResult = (sessionId: string): SessionResumeResult => ({ session_id: sessionId }) as SessionResumeResult

const ports = (over: Partial<ReconnectResumePorts> = {}): ReconnectResumePorts => ({
  dropReplayCursor: vi.fn(),
  healView: vi.fn(),
  resume: vi.fn(async () => resumeResult('rt-fresh')),
  sleep: vi.fn(async () => undefined),
  surfaces: () => [{ runtimeId: 'rt-dead', storedSessionId: 'st-1' }] as LiveResumeSurface[],
  ...over
})

afterEach(() => {
  resetReconnectResumeForTests()
})

// The runtime→stored mapping the follow-up resolves through (the live store does this in the app).
beforeEach(() => {
  $sessionStates.set({ 'rt-dead': { storedSessionId: 'st-1' } as never })
})

describe('reconnect-resume', () => {
  it('resumes every live surface once per epoch, no matter how many waiters ask', async () => {
    const injected = ports({ surfaces: () => [{ storedSessionId: 'st-1' }, { storedSessionId: 'st-2' }] })

    beginReconnectResumeEpoch()

    const gates = [ensureLiveSessionsResumed(injected), ensureLiveSessionsResumed(injected)]

    await Promise.all(gates)

    expect(injected.resume).toHaveBeenCalledTimes(2)
    expect(gates[0]).toBe(gates[1])
  })

  it('does nothing without an armed epoch', async () => {
    const injected = ports()

    await ensureLiveSessionsResumed(injected)

    expect(injected.resume).not.toHaveBeenCalled()
    await expect(reconnectFollowUpRuntimeId('rt-dead')).resolves.toBe('rt-dead')
  })

  it('retries once, after the settling delay, when the gateway is still settling a disconnect', async () => {
    const injected = ports({
      resume: vi
        .fn<ReconnectResumePorts['resume']>()
        .mockRejectedValueOnce(gatewayError(4009))
        .mockResolvedValueOnce(resumeResult('rt-fresh'))
    })

    beginReconnectResumeEpoch()
    await ensureLiveSessionsResumed(injected)

    expect(injected.resume).toHaveBeenCalledTimes(2)
    expect(injected.sleep).toHaveBeenCalledWith(RECONNECT_RESUME_SETTLING_RETRY_MS)
    await expect(reconnectFollowUpRuntimeId('rt-dead')).resolves.toBe('rt-fresh')
  })

  it('never loops: a second settling refusal is the verdict', async () => {
    const injected = ports({ resume: vi.fn(async () => Promise.reject(gatewayError(4009))) })

    beginReconnectResumeEpoch()
    await ensureLiveSessionsResumed(injected)

    expect(injected.resume).toHaveBeenCalledTimes(2)
    await expect(reconnectFollowUpRuntimeId('rt-dead')).resolves.toBe('rt-dead')
  })

  it.each([
    ['the legacy 4007', () => gatewayError(4007)],
    ['4001', () => gatewayError(4001)],
    ['the canonical not_found reason', () => gatewayError(4001, 'not_found')]
  ])('drops the dead cursor and hands the surface back on %s', async (_label, make) => {
    const injected = ports({ resume: vi.fn(async () => Promise.reject(make())) })

    beginReconnectResumeEpoch()
    await ensureLiveSessionsResumed(injected)

    expect(injected.resume).toHaveBeenCalledTimes(1)
    expect(injected.dropReplayCursor).toHaveBeenCalledWith('rt-dead')
    expect(injected.healView).toHaveBeenCalledWith('rt-dead')
    // A gone session is nobody's follow-up: the stored-session affordance owns it.
    await expect(reconnectFollowUpRuntimeId('rt-dead')).resolves.toBeNull()
  })

  it('leaves the caller on its own runtime id when the resume fails transiently', async () => {
    const injected = ports({ resume: vi.fn(async () => Promise.reject(new Error('transport blip'))) })

    beginReconnectResumeEpoch()
    await ensureLiveSessionsResumed(injected)

    expect(injected.resume).toHaveBeenCalledTimes(1)
    expect(injected.dropReplayCursor).not.toHaveBeenCalled()
    await expect(reconnectFollowUpRuntimeId('rt-dead')).resolves.toBe('rt-dead')
  })

  it('a later epoch re-resumes the surfaces again', async () => {
    const injected = ports()

    beginReconnectResumeEpoch()
    await ensureLiveSessionsResumed(injected)
    beginReconnectResumeEpoch()
    await ensureLiveSessionsResumed(injected)

    expect(injected.resume).toHaveBeenCalledTimes(2)
  })
})

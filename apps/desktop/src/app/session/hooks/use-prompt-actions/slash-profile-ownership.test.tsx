import { cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { prepareDefaultNewSession } from '@/app/session/new-session-route'
import { getProfiles } from '@/hermes'
import { $defaultProfileRoute } from '@/store/default-profile'
import {
  $newChatProfile,
  captureNewChatSource,
  pinNewChatProfile
} from '@/store/profile'
import { $profileSwitchFailure, invalidateProfileSwitch, trackProfileSwitch } from '@/store/profile-switch-state'
import { $connection } from '@/store/session'
import { dropSessionState } from '@/store/session-states'

import { actRender, Harness, type HarnessHandle, RUNTIME_SESSION_ID } from './index-test-utils'

const fixture = vi.hoisted(() => ({ activate: vi.fn<() => Promise<void>>() }))
vi.mock('@/hermes', () => ({
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  getProfiles: vi.fn(async () => ({ profiles: [{ name: 'alpha' }, { name: 'beta' }] })),
  getHermesConfig: vi.fn(async () => ({ display: { timestamps: false } })),
  getHermesConfigDefaults: vi.fn(async () => ({})),
  getSession: vi.fn(),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn(),
  transcribeAudio: vi.fn()
}))
vi.mock('@/store/profile', async original => ({
  ...(await original<Record<string, unknown>>()),
  ensureGatewayProfile: fixture.activate
}))

afterEach(() => {
  cleanup()
  $connection.set(null)
  $newChatProfile.set(null)
  captureNewChatSource(null)
  dropSessionState(RUNTIME_SESSION_ID)
  vi.restoreAllMocks()
})

it('keeps the newer profile failure retryable when an older slash activation finishes', async () => {
  invalidateProfileSwitch()
  $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as never)
  const alpha = Promise.withResolvers<void>()
  const beta = Promise.withResolvers<void>()
  fixture.activate.mockReturnValue(alpha.promise)
  let handle: HarnessHandle | null = null
  await actRender(<Harness onReady={value => { handle = value }} refreshSessions={async () => undefined} requestGateway={vi.fn(async () => ({} as never))} />)
  const slash = handle!.submitText('/profile alpha')
  await vi.waitFor(() => expect(fixture.activate).toHaveBeenCalledWith('alpha'))
  pinNewChatProfile('beta')
  const retry = vi.fn()
  const tracked = trackProfileSwitch('beta', beta.promise, retry).catch(() => undefined)
  alpha.resolve()
  await slash
  beta.reject(new Error('beta unavailable'))
  await tracked
  expect($newChatProfile.get()).toBe('beta')
  expect($profileSwitchFailure.get()?.profile).toBe('beta')
  $profileSwitchFailure.get()!.retry()
  expect(retry).toHaveBeenCalledOnce()
})

it('does not rehome a newer generic draft when an older profile lookup resolves', async () => {
  $defaultProfileRoute.set(null)
  pinNewChatProfile('beta')
  fixture.activate.mockReset().mockResolvedValue(undefined)
  const lookup = Promise.withResolvers<Awaited<ReturnType<typeof getProfiles>>>()
  vi.mocked(getProfiles).mockReturnValueOnce(lookup.promise)
  let handle: HarnessHandle | null = null
  await actRender(<Harness onReady={value => { handle = value }} refreshSessions={async () => undefined} requestGateway={vi.fn(async () => ({} as never))} />)
  const callsBefore = vi.mocked(getProfiles).mock.calls.length
  const slash = handle!.submitText('/profile alpha')
  await vi.waitFor(() => expect(vi.mocked(getProfiles).mock.calls.length).toBe(callsBefore + 1))
  prepareDefaultNewSession()
  lookup.resolve({ profiles: [{ name: 'alpha' }] } as Awaited<ReturnType<typeof getProfiles>>)
  await slash
  expect($newChatProfile.get()).toBe('beta')
  expect(fixture.activate).not.toHaveBeenCalled()
})

it.each([true, false])('keeps the latest slash selection when lookups overlap (older first: %s)', async olderFirst => {
  fixture.activate.mockReset().mockResolvedValue(undefined)
  let handle: HarnessHandle | null = null
  await actRender(<Harness onReady={value => { handle = value }} refreshSessions={async () => undefined} requestGateway={vi.fn(async () => ({} as never))} />)
  const alpha = Promise.withResolvers<Awaited<ReturnType<typeof getProfiles>>>()
  const beta = Promise.withResolvers<Awaited<ReturnType<typeof getProfiles>>>()
  vi.mocked(getProfiles).mockReturnValueOnce(alpha.promise).mockReturnValueOnce(beta.promise)
  const first = handle!.submitTextRaw('/profile alpha')
  const second = handle!.submitTextRaw('/profile beta')
  const profiles = { profiles: [{ name: 'alpha' }, { name: 'beta' }] } as Awaited<ReturnType<typeof getProfiles>>

  if (olderFirst) {
    alpha.resolve(profiles)
    await first
    beta.resolve(profiles)
  } else {
    beta.resolve(profiles)
    await second
    alpha.resolve(profiles)
  }

  await Promise.all([first, second])
  expect($newChatProfile.get()).toBe('beta')
  expect(fixture.activate.mock.calls).toEqual([['beta']])
})

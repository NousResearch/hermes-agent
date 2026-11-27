import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { OAuthProvider } from '@/types/hermes'

import {
  $voiceSignIn,
  cancelVoiceSignIn,
  refreshVoiceSignIn,
  signOutVoiceProvider,
  startVoiceSignIn,
  voiceSignInState
} from './voice-engine-sign-in'

const api = vi.hoisted(() => ({
  cancelOAuthSession: vi.fn(async () => ({ ok: true })),
  disconnectOAuthProvider: vi.fn(async () => ({ ok: true, provider: 'xai-oauth' })),
  listOAuthProviders: vi.fn(),
  pollOAuthSession: vi.fn(),
  startOAuthLogin: vi.fn()
}))

vi.mock('@/hermes', () => ({ ...api, OAUTH_LONG_POLL_SECONDS: 25 }))
vi.mock('@/lib/long-poll', () => ({ restOfLongPollGap: async () => undefined }))

const TOKEN = 'xai-SECRET-TOKEN-PREVIEW'

function provider(loggedIn: boolean, extra: Partial<OAuthProvider> = {}): OAuthProvider {
  return {
    cli_command: 'hermes auth add xai-oauth',
    docs_url: '',
    flow: 'device_code',
    id: 'xai-oauth',
    name: 'xAI Grok',
    status: { logged_in: loggedIn, token_preview: loggedIn ? TOKEN : null },
    ...extra
  }
}

const deviceStart = {
  expires_in: 900,
  flow: 'device_code' as const,
  poll_interval: 5,
  session_id: 'sess-1',
  user_code: 'ABCD-1234',
  verification_url: 'https://accounts.x.ai/device'
}

beforeEach(() => {
  $voiceSignIn.set({})
  api.listOAuthProviders.mockResolvedValue({ providers: [provider(false)] })
})

afterEach(() => vi.clearAllMocks())

describe('voice engine sign-in', () => {
  it('reads only logged_in and the name from the provider list, never the token preview', async () => {
    api.listOAuthProviders.mockResolvedValue({ providers: [provider(true)] })

    const state = await refreshVoiceSignIn('xai-oauth', 'work')

    expect(api.listOAuthProviders).toHaveBeenCalledWith('work')
    expect(state).toMatchObject({ disconnectable: true, loggedIn: true, name: 'xAI Grok' })
    expect(JSON.stringify($voiceSignIn.get())).not.toContain(TOKEN)
  })

  it('treats an unknown provider as unknown, not as signed out', async () => {
    api.listOAuthProviders.mockResolvedValue({ providers: [] })

    expect((await refreshVoiceSignIn('xai-oauth')).loggedIn).toBeNull()
  })

  it('starts the device-code flow, opens the page, shows the code, and long-polls to approval', async () => {
    api.startOAuthLogin.mockResolvedValue(deviceStart)
    const seen: string[] = []

    api.pollOAuthSession
      .mockImplementationOnce(async () => {
        const { flow } = voiceSignInState('xai-oauth')
        seen.push(flow.status === 'pending' ? flow.userCode : flow.status)

        return { session_id: 'sess-1', status: 'pending' }
      })
      .mockImplementationOnce(async () => {
        api.listOAuthProviders.mockResolvedValue({ providers: [provider(true)] })

        return { session_id: 'sess-1', status: 'approved' }
      })

    const openUrl = vi.fn()

    await startVoiceSignIn('xai-oauth', undefined, openUrl)

    expect(openUrl).toHaveBeenCalledWith('https://accounts.x.ai/device')
    expect(seen).toEqual(['ABCD-1234'])
    expect(api.pollOAuthSession).toHaveBeenCalledWith('xai-oauth', 'sess-1', undefined, 25)
    expect(voiceSignInState('xai-oauth')).toMatchObject({ flow: { status: 'idle' }, loggedIn: true })
    expect(JSON.stringify($voiceSignIn.get())).not.toContain(TOKEN)
  })

  it('surfaces a denied or expired sign-in as an error', async () => {
    api.startOAuthLogin.mockResolvedValue(deviceStart)
    api.pollOAuthSession.mockResolvedValue({
      error_message: 'The code expired',
      session_id: 'sess-1',
      status: 'expired'
    })

    await startVoiceSignIn('xai-oauth', undefined, vi.fn())

    expect(voiceSignInState('xai-oauth').flow).toEqual({ message: 'The code expired', status: 'error' })
  })

  it('cancel stops the poll loop and cancels the backend session', async () => {
    api.startOAuthLogin.mockResolvedValue(deviceStart)
    let releasePoll: (value: unknown) => void = () => undefined
    api.pollOAuthSession.mockImplementation(() => new Promise(resolve => (releasePoll = resolve)))

    const running = startVoiceSignIn('xai-oauth', 'work', vi.fn())
    await vi.waitFor(() => expect(voiceSignInState('xai-oauth', 'work').flow.status).toBe('pending'))

    await cancelVoiceSignIn('xai-oauth', 'work')
    expect(api.cancelOAuthSession).toHaveBeenCalledWith('sess-1', 'work')

    // A late "approved" answer for the abandoned session must not flip the state.
    releasePoll({ session_id: 'sess-1', status: 'approved' })
    await running

    expect(voiceSignInState('xai-oauth', 'work')).toMatchObject({ flow: { status: 'idle' }, loggedIn: null })
    expect(api.pollOAuthSession).toHaveBeenCalledTimes(1)
  })

  it('sign-out removes the provider sign-in for that profile and re-reads the state', async () => {
    await signOutVoiceProvider('xai-oauth', 'work')

    expect(api.disconnectOAuthProvider).toHaveBeenCalledWith('xai-oauth', 'work')
    expect(voiceSignInState('xai-oauth', 'work').loggedIn).toBe(false)
  })

  it('keeps sign-in state per profile', async () => {
    api.listOAuthProviders.mockImplementation(async (profile?: string) => ({
      providers: [provider(profile === 'work')]
    }))

    await refreshVoiceSignIn('xai-oauth', 'work')
    await refreshVoiceSignIn('xai-oauth')

    expect(voiceSignInState('xai-oauth', 'work').loggedIn).toBe(true)
    expect(voiceSignInState('xai-oauth').loggedIn).toBe(false)
  })
})

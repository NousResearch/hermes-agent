import { map } from 'nanostores'

import {
  cancelOAuthSession,
  disconnectOAuthProvider,
  listOAuthProviders,
  OAUTH_LONG_POLL_SECONDS,
  pollOAuthSession,
  startOAuthLogin
} from '@/hermes'
import { restOfLongPollGap } from '@/lib/long-poll'

/**
 * Sign-in state for the Hermes OAuth provider a live voice engine declared
 * (`LiveVoiceEngine.signIn.provider`). The flow is the one onboarding and the
 * toolset panel already drive (`/api/providers/oauth/<id>/start` → open the
 * verification page → long-poll), so the engine's plugin never sees a token:
 * its backend reaches the provider through `credentialed_provider_request`.
 *
 * Only `logged_in` and the display name are read from the provider status;
 * `token_preview` and every other credential field are dropped here.
 */
export type VoiceSignInFlow =
  | { status: 'idle' }
  | { status: 'starting' }
  | { sessionId: string; status: 'pending'; userCode: string; verificationUrl: string }
  | { message: string; status: 'error' }

export interface VoiceSignInState {
  /** Null until the provider list answered (or when the provider is unknown). */
  loggedIn: boolean | null
  /** The provider's display name, or its id until the list answered. */
  name: string
  /** False when the backend says this provider cannot be signed out from the app. */
  disconnectable: boolean
  flow: VoiceSignInFlow
}

/** Keyed by `signInKey(provider, profile)`: a sign-in belongs to one profile. */
export const $voiceSignIn = map<Record<string, VoiceSignInState>>({})

export const signInKey = (provider: string, profile?: null | string) => `${profile ?? ''}\u0000${provider}`

// Bumped per start/cancel so a superseded long-poll loop stops publishing.
const generations = new Map<string, number>()

const unknownState = (provider: string): VoiceSignInState => ({
  disconnectable: true,
  flow: { status: 'idle' },
  loggedIn: null,
  name: provider
})

function update(key: string, provider: string, patch: Partial<VoiceSignInState>) {
  const current = $voiceSignIn.get()[key] ?? unknownState(provider)

  $voiceSignIn.setKey(key, { ...current, ...patch })
}

export function voiceSignInState(provider: string, profile?: null | string): VoiceSignInState {
  return $voiceSignIn.get()[signInKey(provider, profile)] ?? unknownState(provider)
}

/** Re-read whether `provider` is signed in for `profile`. */
export async function refreshVoiceSignIn(provider: string, profile?: null | string): Promise<VoiceSignInState> {
  const key = signInKey(provider, profile)

  try {
    const { providers } = await listOAuthProviders(profile ?? undefined)
    const entry = providers.find(candidate => candidate.id === provider)

    update(key, provider, {
      disconnectable: entry?.disconnectable !== false,
      loggedIn: entry ? Boolean(entry.status?.logged_in) : null,
      name: entry?.name || provider
    })
  } catch {
    update(key, provider, { loggedIn: null })
  }

  return voiceSignInState(provider, profile)
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error))

/** Start the provider's device-code sign-in: publish the code, open the
 *  verification page, then long-poll until it resolves. `openUrl` is the
 *  caller's door to the OS browser. */
export async function startVoiceSignIn(
  provider: string,
  profile: null | string | undefined,
  openUrl: (url: string) => Promise<unknown> | void
): Promise<void> {
  const key = signInKey(provider, profile)
  const generation = (generations.get(key) ?? 0) + 1
  generations.set(key, generation)
  const live = () => generations.get(key) === generation

  update(key, provider, { flow: { status: 'starting' } })

  try {
    const start = await startOAuthLogin(provider, profile ?? undefined)

    if (!live()) {
      return
    }

    if (start.flow !== 'device_code') {
      update(key, provider, { flow: { message: `unsupported sign-in flow: ${start.flow}`, status: 'error' } })

      return
    }

    update(key, provider, {
      flow: {
        sessionId: start.session_id,
        status: 'pending',
        userCode: start.user_code,
        verificationUrl: start.verification_url
      }
    })
    void Promise.resolve(openUrl(start.verification_url)).catch(() => undefined)

    for (;;) {
      const askedAt = Date.now()
      const result = await pollOAuthSession(provider, start.session_id, profile ?? undefined, OAUTH_LONG_POLL_SECONDS)

      if (!live()) {
        return
      }

      if (result.status === 'approved') {
        update(key, provider, { flow: { status: 'idle' }, loggedIn: true })
        await refreshVoiceSignIn(provider, profile)

        return
      }

      if (result.status !== 'pending') {
        update(key, provider, { flow: { message: result.error_message || result.status, status: 'error' } })

        return
      }

      await restOfLongPollGap(askedAt)

      if (!live()) {
        return
      }
    }
  } catch (error) {
    if (live()) {
      update(key, provider, { flow: { message: errorText(error), status: 'error' } })
    }
  }
}

/** Abandon a pending sign-in (the backend session is cancelled too). */
export async function cancelVoiceSignIn(provider: string, profile?: null | string): Promise<void> {
  const key = signInKey(provider, profile)
  const { flow } = voiceSignInState(provider, profile)
  generations.set(key, (generations.get(key) ?? 0) + 1)
  update(key, provider, { flow: { status: 'idle' } })

  if (flow.status === 'pending') {
    await cancelOAuthSession(flow.sessionId, profile ?? undefined).catch(() => undefined)
  }
}

/** Remove the provider's stored sign-in for `profile`. Throws on failure so
 *  the caller can say so. */
export async function signOutVoiceProvider(provider: string, profile?: null | string): Promise<void> {
  await disconnectOAuthProvider(provider, profile ?? null)
  update(signInKey(provider, profile), provider, { flow: { status: 'idle' }, loggedIn: false })
  await refreshVoiceSignIn(provider, profile)
}

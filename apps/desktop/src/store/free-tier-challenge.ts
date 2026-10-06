import type { FreeTierChallengePayload, FreeTierChallengeResultParams } from '@hermes/shared'

/**
 * The free tier's browser challenge, renderer half: a relay. The backend
 * (`hermes_cli/anon_challenge.py`) announces a challenge and polls the account
 * service for the result itself; the Electron main process
 * (`electron/challenge-window.ts`) loads the page hidden and decides when to
 * reveal it. This relays each presentation attempt and reports how the window
 * ended, once, so the backend can re-mint promptly instead of polling out.
 * The event and a status read can both name the same attempt.
 */

export type FreeTierChallengeOutcome = FreeTierChallengeResultParams['outcome']

/** Sends ``free_tier.challenge_result`` to the backend that announced the challenge. */
export type ChallengeRequester = (method: 'free_tier.challenge_result', params: Record<string, unknown>) => Promise<unknown>

const inFlight = new Map<string, Promise<FreeTierChallengeOutcome>>()

function isBrowserChallenge(value: unknown): value is FreeTierChallengePayload {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const candidate = value as Partial<FreeTierChallengePayload>

  return candidate.type === 'browser' && typeof candidate.url === 'string' && candidate.url.length > 0
}

export function runFreeTierChallenge(
  challenge: unknown,
  requestGateway?: ChallengeRequester
): Promise<FreeTierChallengeOutcome> | null {
  if (!isBrowserChallenge(challenge)) {
    return null
  }

  const attempt = challenge.attempt ?? 0
  const key = `${challenge.url}:${attempt}`
  const existing = inFlight.get(key)

  if (existing) {
    // The first asker's report covers this attempt; a second report would
    // only repeat it.
    return existing
  }

  // A hint, not a verdict: an older backend without the RPC, or a dropped
  // socket, loses it and falls back to its bounded status poll. The result
  // never grants a credential, so a failed report is not worth surfacing.
  const reported = async (outcome: FreeTierChallengeOutcome) => {
    try {
      await requestGateway?.('free_tier.challenge_result', { url: challenge.url, attempt, outcome })
    } catch {
      // see above
    }

    return outcome
  }

  const bridge = window.hermesDesktop?.freeTierChallenge

  if (!bridge) {
    // A web build, or a desktop shell older than this renderer: nothing can
    // host the page. Say so, so the backend can end its wait promptly.
    return reported('unsupported')
  }

  const run = bridge
    .run({ url: challenge.url, required: challenge.required !== false, expiresIn: challenge.expires_in, attempt })
    .catch((): FreeTierChallengeOutcome => 'error')
    .then(reported)
    .finally(() => inFlight.delete(key))

  inFlight.set(key, run)

  return run
}

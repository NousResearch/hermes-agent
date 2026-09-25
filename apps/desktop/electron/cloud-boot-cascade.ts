/**
 * cloud-boot-cascade.ts
 *
 * Pure decision seam for self-healing a Hermes Cloud agent connection at boot.
 *
 * A `cloud` connection authenticates to its agent with a bearer minted by the
 * silent per-agent token exchange (main.ts `cloudAgentAuth.signIn`, audience
 * taken only from a portal-confirmed binding). When the gateway rejects the
 * stored bearer, the WS-ticket mint answers 401 and boot would latch reauth,
 * even though the desktop portal session that can mint a fresh bearer is
 * still live.
 *
 * This module decides when the boot path may run one silent sign-in and
 * retry. Kept free of `electron` imports so it unit-tests in the electron
 * vitest project; main.ts owns the side effects.
 */

import { isReauthRequiredError } from './backend-health'

export interface CloudBootCascadeCandidate {
  remoteKind?: unknown
  authMode?: unknown
}

/**
 * True when a failed `waitForHermes` for `remote` should be followed by one
 * silent per-agent sign-in and a single retry, rather than surfacing the
 * reauth error immediately. Requires all of:
 *
 *   - the connection is a Hermes Cloud agent (`remoteKind: 'cloud'`);
 *   - it uses the OAuth auth mode (`authMode: 'oauth'`), the only mode the
 *     token exchange can mint credentials for;
 *   - the failure is the terminal reauth error (`isReauthRequired`), i.e. the
 *     ticket mint rejected the session. Transport errors, server-side 5xx and
 *     anything else keep their existing handling.
 *
 * The caller must additionally confirm a live portal session before running
 * the sign-in; without one the exchange cannot succeed and would only add a
 * delay in front of the same error.
 */
export function shouldAttemptCloudBootCascade(remote: CloudBootCascadeCandidate | null | undefined, error: unknown): boolean {
  if (!remote || typeof remote !== 'object') {
    return false
  }

  if (remote.remoteKind !== 'cloud') {
    return false
  }

  if (remote.authMode !== 'oauth') {
    return false
  }

  return isReauthRequiredError(error)
}

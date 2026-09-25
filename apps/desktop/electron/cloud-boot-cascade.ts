/**
 * cloud-boot-cascade.ts
 *
 * Pure decision seam for self-healing a Hermes Cloud agent connection at boot.
 *
 * A `cloud` connection authenticates to its agent through the silent per-agent
 * cascade (main.ts `cloudAgentSilentSignIn`): open the agent's protected root in
 * the shared OAuth partition, let the portal auto-approve, and the agent's own
 * session cookie lands with no prompt. That cascade was only ever driven by the
 * settings UI ("Use gateway"). The boot path went straight to `waitForHermes`,
 * so once the agent cookie expired the WS-ticket mint answered 401, the app
 * reported "not signed in" and latched reauth, even though the portal session
 * it needed to recover was still live. Every relaunch needed a manual click.
 *
 * This module decides when the boot path may run the cascade once and retry.
 * Kept free of `electron` imports so it unit-tests in the electron vitest
 * project; main.ts owns the side effects.
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
 *   - it authenticates with cookies (`authMode: 'oauth'`), which is the only
 *     mode the cascade can mint a session for;
 *   - the failure is the terminal reauth error (`isReauthRequired`), i.e. the
 *     ticket mint rejected the session. Transport errors, server-side 5xx and
 *     anything else keep their existing handling.
 *
 * The caller must additionally confirm a live portal session before running
 * the cascade; without one the cascade cannot succeed and would only add a
 * hidden window and a delay in front of the same error.
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

/**
 * remote-only-local-bootstrap.ts
 *
 * "Only use my remote gateway" enforcement for the LOCAL installer (#112514).
 *
 * The first-run setup screen already offers Connect to existing Hermes instead
 * of a local install, and applying it persists that pick as this Desktop's
 * primary connection (connection.json `mode`, or a remote/cloud/ssh registry
 * primary). What was missing was honoring the pick everywhere a local install
 * can still start: the pooled-profile spawn and the repair/re-resolve paths
 * call runEnsureRuntime() directly, which drove install.ps1/install.sh on a
 * machine whose owner had already chosen a remote gateway — the "local install
 * I will never use" in the report.
 *
 * Kept electron-free and pure: main.ts supplies the answer to "is this
 * Desktop's primary connection a remote gateway?" (its existing
 * globalRemoteActive(), which already folds in the registry primary), so the
 * decision is unit-testable without booting Electron.
 */

/** Copy for a skipped local install. Plain, actionable, no jargon. */
export const REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE =
  'No local Hermes install was started because this Desktop is set to your remote gateway. ' +
  'To install Hermes on this computer instead, switch Settings → Gateway to "This device".'

/** Typed so callers can tell a deliberate skip from a failed install. */
export class RemoteOnlyLocalBootstrapError extends Error {
  readonly localBootstrapSkipped = true

  constructor(message = REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE) {
    super(message)
    this.name = 'RemoteOnlyLocalBootstrapError'
  }
}

/**
 * Why a local install must not start, or null when it may.
 *
 * `remotePrimary` is the persisted choice, re-read on every call — a Desktop
 * whose primary connection is a remote gateway never installs the local
 * runtime behind the user's back, and one set to local is untouched.
 */
export function remoteOnlyLocalBootstrapReason(remotePrimary: boolean): null | string {
  return remotePrimary ? REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE : null
}

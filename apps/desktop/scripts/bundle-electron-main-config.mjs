/**
 * Compile-time defines for the Electron entry bundles.
 *
 * Production builds always define the remote-only capability explicitly.  An
 * absent define would leave `process.env.HERMES_DESKTOP_REMOTE_ONLY` mutable at
 * runtime, allowing a normal Desktop install to be turned into a different
 * product by an inherited environment variable.
 */
export function electronBundleDefines({ isDev = false, isRemoteOnly = false } = {}) {
  if (isDev) {
    // Developer bundles retain the existing environment-driven harness unless
    // the developer explicitly requests the standalone flavor.
    return isRemoteOnly ? { 'process.env.HERMES_DESKTOP_REMOTE_ONLY': JSON.stringify('1') } : {}
  }

  return {
    'process.env.HERMES_DESKTOP_IS_PACKAGED': JSON.stringify(true),
    'process.env.HERMES_DESKTOP_REMOTE_ONLY': JSON.stringify(isRemoteOnly ? '1' : '0')
  }
}

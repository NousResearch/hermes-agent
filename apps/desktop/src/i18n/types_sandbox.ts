/** Copy for a profile a plugin runs in a sandbox on this computer (host.setProfileBackend). */
export interface SandboxCopy {
  /** While the plugin starts the profile's sandbox (this computer's backend keeps serving it). */
  starting: string
  /** While the sandbox stops and the profile moves back to this computer's backend. */
  stopping: string
  /** A planned restart of the sandboxed backend. */
  restarting: string
}

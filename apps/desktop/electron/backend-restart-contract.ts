export interface BackendRestartTarget {
  connectionId: string
  profile: string
}

export type BackendRestartUnsupportedReason =
  | 'invalid-target'
  | 'unknown-connection'
  | 'externally-managed'
  | 'ssh-ownership-unverified'
  | 'not-owned'
  | 'target-changed'

export type BackendRestartCapability =
  { supported: true } | { supported: false; reason: BackendRestartUnsupportedReason }

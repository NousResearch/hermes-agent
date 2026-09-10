/** Reserved text frames are never terminal input, even when malformed. */
export const DRAFT_CONTROL_PREFIX = '\u0000hermes-draft-v1:'

export interface DraftIdentity {
  pty_instance: string
  connection_generation: number
  session_id: string
  draft_id: string
}

export interface DraftState {
  type: 'draft.state'
  identity: DraftIdentity
  available: boolean
}

export interface DraftAttachRequest {
  type: 'draft.attach'
  request_id: string
  expected: DraftIdentity
  path: string
}

export interface DraftResult {
  type: 'draft.result'
  request_id: string
  identity: DraftIdentity
  status: 'attached' | 'stale' | 'unavailable' | 'failed'
  error?: string
  path?: string
  label?: string
}

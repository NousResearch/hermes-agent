export interface SessionCreateOverrides {
  /** Renderer-only handoff, fired at the stored-id assignment before navigation. */
  onComposerScopeAssigned?: (scope: string) => void
  /** Created after a chat switch: deliver the first turn without stealing focus. */
  onCreatedInBackground?: (storedSessionId: string) => void
}

export type CreateBackendSessionForSend = (
  preview?: string | null,
  createOverrides?: SessionCreateOverrides
) => Promise<string | null>

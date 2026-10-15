export interface SessionCreateOverrides {
  /** Renderer-only handoff, fired at the stored-id assignment before navigation. */
  onComposerScopeAssigned?: (scope: string) => void
}

export type CreateBackendSessionForSend = (
  preview?: string | null,
  createOverrides?: SessionCreateOverrides,
  draftKey?: string | null
) => Promise<string | null>

/** Electron's process singleton is acquired before any shared rollout state opens. */
export interface ManagedRolloutSingleInstanceApp {
  requestSingleInstanceLock(): boolean
  hasSingleInstanceLock(): boolean
}

export interface ManagedRolloutProcessOwner {
  readonly acquired: boolean
  owns(): boolean
  assert(): void
}

export function acquireManagedRolloutProcessOwner(app: ManagedRolloutSingleInstanceApp): ManagedRolloutProcessOwner {
  const acquired = app.requestSingleInstanceLock()
  const owns = () => acquired && app.hasSingleInstanceLock()

  return {
    acquired,
    owns,
    assert: () => {
      if (!owns()) {throw new Error('managed-update-owner-unavailable')}
    }
  }
}

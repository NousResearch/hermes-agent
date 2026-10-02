import { waitForTeardown } from './local-backend-lifecycle'

/** Retain transports removed from routing until their remote kill has settled. */
export function createSshTeardownTracker() {
  const pending = new Map<Promise<void>, { close: () => Promise<unknown> }>()

  return {
    hasPending: () => pending.size > 0,
    track(ssh: { close: () => Promise<unknown> }, teardown: () => Promise<void>): Promise<void> {
      const promise = Promise.resolve().then(teardown)
      pending.set(promise, ssh)
      void promise.then(
        () => pending.delete(promise),
        () => pending.delete(promise)
      )

      return promise
    },
    async finish(
      bootstraps: Promise<unknown>[],
      forceCleanup: () => Promise<unknown>,
      graceMs = 6_000,
      forceMs = 1_000
    ) {
      await waitForTeardown([...pending.keys(), ...bootstraps], graceMs)
      // Only abandon the transport after the existing remote-kill grace period.
      // Closing SSH unblocks a wedged exec; both close and bootstrap force cleanup
      // are bounded too, not an unbounded await after a nominal timeout.
      await waitForTeardown(
        [
          Promise.resolve().then(forceCleanup),
          ...[...new Set(pending.values())].map(ssh => Promise.resolve().then(() => ssh.close()))
        ],
        forceMs
      )
    }
  }
}

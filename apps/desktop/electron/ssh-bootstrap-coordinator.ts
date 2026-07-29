import crypto from 'node:crypto'

function sshConfigFingerprint(scope, config) {
  const parts = [
    scope,
    config.host,
    config.user,
    config.port,
    config.keyPath,
    config.remoteHermesPath,
    config.effectiveConfigFingerprint
  ]

  return crypto
    .createHash('sha256')
    .update(JSON.stringify(parts.map(value => value ?? '')))
    .digest('hex')
}

function createBootstrapCoordinator() {
  const active = new Set<any>()
  const pending = new Map<string, any>()
  const generations = new Map<string, number>()
  const drains = new Map<string, Promise<void>>()

  function start(scope, fingerprint, run) {
    const current = pending.get(scope)

    if (current?.fingerprint === fingerprint) {
      return current.promise
    }

    current?.controller.abort()

    const generation = (generations.get(scope) || 0) + 1
    generations.set(scope, generation)
    const controller = new AbortController()
    const forceCleanups = new Set<() => any>()

    const lease = {
      signal: controller.signal,
      onForceCleanup(cleanup) {
        forceCleanups.add(cleanup)

        return () => forceCleanups.delete(cleanup)
      },
      isCurrent: () => !controller.signal.aborted && generations.get(scope) === generation,
      assertCurrent() {
        if (!this.isCurrent()) {
          const error: any = new Error('SSH bootstrap was superseded by newer connection settings.')
          error.kind = 'superseded'
          throw error
        }
      }
    }

    const drain = drains.get(scope) || Promise.resolve()
    const predecessor = current ? Promise.allSettled([current.promise, drain]) : drain
    const entry: any = { controller, fingerprint, forceCleanups, generation, promise: null, scope }

    const promise = predecessor
      .then(() => {
        lease.assertCurrent()

        return run(lease)
      })
      .finally(() => {
        forceCleanups.clear()
        active.delete(entry)

        if (pending.get(scope)?.generation === generation) {
          pending.delete(scope)
        }
      })

    entry.promise = promise
    active.add(entry)
    pending.set(scope, entry)

    return promise
  }

  function cancel(scope) {
    pending.get(scope)?.controller.abort()
  }

  async function cancelAndWait(scope) {
    let release

    const barrier = new Promise<void>(resolve => {
      release = resolve
    })

    drains.set(scope, barrier)
    const entries = [...active].filter(entry => entry.scope === scope)

    for (const entry of entries) {
      entry.controller.abort()
    }

    try {
      await Promise.allSettled(entries.map(entry => entry.promise))
    } finally {
      if (drains.get(scope) === barrier) {
        drains.delete(scope)
      }

      release()
    }
  }

  function cancelAll() {
    for (const entry of active) {
      entry.controller.abort()
    }
  }

  async function forceCleanupAll() {
    const cleanups = [...active].flatMap(entry => [...entry.forceCleanups])
    await Promise.allSettled(cleanups.map(cleanup => cleanup()))
  }

  function promises() {
    return [...active].map(entry => entry.promise)
  }

  return { active, cancel, cancelAll, cancelAndWait, forceCleanupAll, pending, promises, start }
}

export { createBootstrapCoordinator, sshConfigFingerprint }

/**
 * windows-update-orchestration.ts
 *
 * One thing: the updater handoff as a transaction. Spawning a detached updater
 * is a commit point — after it, this Desktop is no longer the owner of its own
 * install — so the spawn, the liveness observation, the authentication, the
 * commit and the compensating restore have to be sequenced in one place rather
 * than re-derived by each caller.
 *
 * Deliberately free of Electron, filesystem and process imports: the callers
 * (main.ts for bootstrap recovery, windows-update-apply.ts for the in-app
 * update) supply every effect, so the ordering is provable on fakes.
 */

export interface UpdaterHandoffObservation {
  ok: boolean
  message?: string
}

export type UpdaterHandoffResult<T> = { ok: true; value: T } | { ok: false; error: string; message: string }

export interface UpdaterHandoffTransaction<TChild, TValue> {
  spawn: () => TChild
  observe: (child: TChild) => Promise<UpdaterHandoffObservation>
  authenticate: (child: TChild) => Promise<boolean>
  commit: (child: TChild) => TValue | Promise<TValue>
  restore: (child: TChild | null) => void | Promise<void>
  authenticationError: string
}

export async function runUpdaterHandoffTransaction<TChild, TValue>(
  transaction: UpdaterHandoffTransaction<TChild, TValue>
): Promise<UpdaterHandoffResult<TValue>> {
  let child: TChild | null = null
  let observation: Promise<UpdaterHandoffObservation> | null = null
  let restorationStarted = false

  const restore = async () => {
    if (restorationStarted) {
      return
    }
    restorationStarted = true
    await transaction.restore(child)
  }

  try {
    child = transaction.spawn()
    observation = transaction.observe(child)

    const authenticated = await transaction.authenticate(child)
    const observed = await observation

    if (!authenticated || !observed.ok) {
      await restore()

      return {
        ok: false,
        error: authenticated ? 'updater-spawn-failed' : 'update-handoff-unacknowledged',
        message: authenticated
          ? observed.message || 'Updater process exited before handoff completed.'
          : transaction.authenticationError
      }
    }

    return { ok: true, value: await transaction.commit(child) }
  } catch (error) {
    if (observation) {
      await observation.catch(() => undefined)
    }
    await restore()
    throw error
  }
}

/** A started updater must never fall through to an in-process update on failure. */
export function requireUpdaterHandoff<T>(result: UpdaterHandoffResult<T>): T {
  if (result.ok === false) {
    throw new Error(result.message)
  }

  return result.value
}

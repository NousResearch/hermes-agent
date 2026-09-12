/**
 * pool-idle-reap.ts
 *
 * Idle-reap selection for the desktop backend pool, gated on backend-side
 * work rather than `lastActiveAt` alone.
 *
 * `lastActiveAt` is refreshed only by renderer attention (chat WS open /
 * streaming keepalive) — the main process has no visibility into work a
 * pooled backend does on its own (a cron job, a long agent turn with no
 * window attached). A backend running such work with no renderer attached
 * looks idle by `lastActiveAt` alone and was reaped mid-run, killing the
 * in-flight execution (#108863). `isBusy` gives the reaper a real work
 * signal to check before it acts on the renderer-attention proxy.
 */

export interface PoolIdleEntry {
  lastActiveAt?: null | number
}

/** Pool keys idle beyond `idleMs` by the `lastActiveAt` renderer-attention signal alone. */
export function selectIdleReapCandidates<K>(
  entries: Iterable<[K, PoolIdleEntry]>,
  now: number,
  idleMs: number
): K[] {
  return [...entries].filter(([, entry]) => now - (entry.lastActiveAt || 0) > idleMs).map(([key]) => key)
}

/**
 * Reap every pool entry idle past `idleMs`, but only after `isBusy` confirms
 * the backend itself has no in-flight work. `isBusy` must fail closed — a
 * probe that cannot tell must report busy, since "can't tell" is exactly the
 * previously-unguarded case that let a cron run get reaped mid-flight.
 */
export async function reapIdleBackends<K>(
  entries: Iterable<[K, PoolIdleEntry]>,
  now: number,
  idleMs: number,
  isBusy: (key: K) => Promise<boolean>,
  stop: (key: K) => Promise<void>
): Promise<K[]> {
  const candidates = selectIdleReapCandidates(entries, now, idleMs)
  const reaped: K[] = []

  await Promise.all(
    candidates.map(async key => {
      if (await isBusy(key)) {
        return
      }

      reaped.push(key)
      await stop(key)
    })
  )

  return reaped
}

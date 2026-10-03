import { atom } from 'nanostores'

/** Whether a session's running turn carries the Daybreak program (`session.info.daybreak_active`),
 *  keyed by both its stored and runtime ids. A queued message with an explicit choice may steer
 *  that turn only when it asks for the same program. */
export const $runningDaybreak = atom<Record<string, boolean>>({})

export function setRunningDaybreak(keys: readonly (null | string | undefined)[], active: boolean): void {
  const current = $runningDaybreak.get()
  const ids = keys.filter((key): key is string => Boolean(key))

  if (ids.every(id => Boolean(current[id]) === active)) {
    return
  }

  const next = { ...current }

  for (const id of ids) {
    if (active) {
      next[id] = true
    } else {
      delete next[id]
    }
  }

  $runningDaybreak.set(next)
}

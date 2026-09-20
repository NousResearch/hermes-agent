import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const TRAJECTORY_COLLAPSED_BY_DEFAULT_STORAGE_KEY = 'hermes.desktop.trajectory.collapsedByDefault'

/** Desktop-local presentation preference; shared backend config must not be changed by a single window. */
export const $trajectoryCollapsedByDefault = atom(storedBoolean(TRAJECTORY_COLLAPSED_BY_DEFAULT_STORAGE_KEY, true))

$trajectoryCollapsedByDefault.subscribe(value => persistBoolean(TRAJECTORY_COLLAPSED_BY_DEFAULT_STORAGE_KEY, value))

export function setTrajectoryCollapsedByDefault(value: boolean) {
  $trajectoryCollapsedByDefault.set(value)
}

import { atom } from 'nanostores'

export const $usageDays = atom(30)

export function setUsageDays(days: number) {
  $usageDays.set(Math.max(1, Math.min(365, days)))
}

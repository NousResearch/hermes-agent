import { atom } from 'nanostores'

export const $threadScrolledUp = atom(false)

export function setThreadScrolledUp(value: boolean) {
  if ($threadScrolledUp.get() === value) {
    return
  }

  $threadScrolledUp.set(value)
}

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'

/**
 * Composers a feature has made read-only for a moment, by composer target
 * (`main`, `tile:<id>`): e.g. while a send is held for review, so nothing typed
 * in the meantime can be overwritten when the held draft is restored. Holds are
 * per token: overlapping holders release independently, and a released token
 * can never release someone else's hold.
 */
const $holds = atom<ReadonlyMap<symbol, string>>(new Map())

export function holdComposerInput(target: string): () => void {
  const token = Symbol(target)
  $holds.set(new Map($holds.get()).set(token, target))

  return () => {
    if (!$holds.get().has(token)) {
      return
    }

    const next = new Map($holds.get())
    next.delete(token)
    $holds.set(next)
  }
}

const heldIn = (holds: ReadonlyMap<symbol, string>, target: string) => [...holds.values()].includes(target)

export const composerInputHeld = (target: string): boolean => heldIn($holds.get(), target)

export function useComposerInputHeld(target: string): boolean {
  return heldIn(useStore($holds), target)
}

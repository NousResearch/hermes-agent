import { type ComposerTarget, getActiveComposer } from './focus'

export function composerTargetAtPoint(x: number, y: number): ComposerTarget {
  return (
    (document.elementFromPoint?.(x, y)?.closest<HTMLElement>('[data-composer-target]')?.dataset.composerTarget as
      | ComposerTarget
      | undefined) ?? getActiveComposer()
  )
}

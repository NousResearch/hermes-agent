import { ContribBoundary } from './boundary'
import { useContributions } from './use-contributions'

export interface SlotProps {
  /** Area id whose contributions render inline, in order. */
  area: string
}

/** Renders a bar area: ordered inline items `[...core, ...plugin]`. */
export function Slot({ area }: SlotProps) {
  const items = useContributions(area)

  if (items.length === 0) {
    return null
  }

  return (
    <>
      {items.map(c => (
        <ContribBoundary id={c.id} key={`${c.source ?? 'core'}:${c.id}`} variant="chip">
          {c.render?.()}
        </ContribBoundary>
      ))}
    </>
  )
}

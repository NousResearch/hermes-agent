import { cn } from '@/lib/utils'

const EDGE_CLASSES = {
  bottom: 'inset-x-0 bottom-0 h-[5px]',
  left: 'inset-y-0 left-0 w-[5px]',
  right: 'inset-y-0 right-0 w-[5px]',
  top: 'inset-x-0 top-0 h-[5px]'
} as const

/** Visual drag frame with pointer-active strips only on the exposed 5px ring. */
export function ComposerDragRegion({ dragging }: { dragging: boolean }) {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0"
      data-dragging={dragging ? '' : undefined}
      data-slot="composer-drag-region"
    >
      {(Object.keys(EDGE_CLASSES) as Array<keyof typeof EDGE_CLASSES>).map(edge => (
        <span
          aria-hidden
          className={cn(
            'pointer-events-auto absolute cursor-grab',
            EDGE_CLASSES[edge],
            dragging && 'cursor-grabbing'
          )}
          data-drag-edge={edge}
          key={edge}
        />
      ))}
    </div>
  )
}
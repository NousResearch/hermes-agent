import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import { readableInk } from '@/themes/color'

/** One accent colour as a round swatch; with `onColorChange` it is the custom-colour well. */
export function AccentSwatch({
  active,
  hex,
  name,
  onColorChange,
  onPick
}: {
  active: boolean
  hex: string
  name: string
  onColorChange?: (hex: string) => void
  onPick?: () => void
}) {
  const className = cn(
    'relative inline-flex size-9 items-center justify-center rounded-full border border-foreground/15 transition-transform duration-150',
    !active && 'hover:scale-105'
  )

  const style = {
    background: hex,
    boxShadow: active ? `0 0 0 2px var(--dt-background), 0 0 0 4px ${hex}` : undefined
  }

  return (
    <Tip label={name}>
      {onColorChange ? (
        <label className={cn(className, 'focus-within:outline-2 focus-within:outline-ring')} style={style}>
          <span className="flex" style={{ color: readableInk(hex) }}>
            <Codicon name="add" size="1rem" />
          </span>
          <input
            aria-label={name}
            className="absolute inset-0 size-full cursor-pointer opacity-0"
            onChange={event => onColorChange(event.target.value)}
            type="color"
            value={hex}
          />
        </label>
      ) : (
        <button
          aria-label={name}
          aria-pressed={active}
          className={className}
          onClick={onPick}
          style={style}
          type="button"
        />
      )}
    </Tip>
  )
}

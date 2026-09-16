import { Codicon } from './codicon'

interface ColorSwatchesProps {
  swatches: readonly string[]
  value: null | string
  onChange: (color: null | string) => void
  clearLabel: string
  clearIcon?: string
  swatchLabel?: (color: string) => string
}

// Shared swatch grid + clear row used by the profile rail and the project
// dialog, so color picking looks and behaves identically everywhere.
export function ColorSwatches({
  swatches,
  value,
  onChange,
  clearLabel,
  clearIcon = 'circle-slash',
  swatchLabel
}: ColorSwatchesProps) {
  return (
    <div>
      <div className="grid grid-cols-6 gap-1.5">
        {swatches.map(swatch => (
          <button
            aria-label={swatchLabel?.(swatch) ?? swatch}
            className="size-5 rounded-full transition-transform hover:scale-110"
            key={swatch}
            onClick={() => onChange(swatch)}
            style={{
              backgroundColor: swatch,
              boxShadow: swatch === value ? '0 0 0 2px var(--ui-bg-elevated), 0 0 0 3.5px currentColor' : undefined,
              color: swatch
            }}
            type="button"
          />
        ))}
      </div>
      <button
        className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-md py-1 text-xs text-(--ui-text-tertiary) transition hover:bg-(--ui-control-hover-background) hover:text-foreground"
        onClick={() => onChange(null)}
        type="button"
      >
        <Codicon name={clearIcon} size="0.75rem" />
        {clearLabel}
      </button>
    </div>
  )
}

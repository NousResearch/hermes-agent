export interface InteriorCutawayToggleProps {
  buildingId?: string
  title?: string
  enabled: boolean
  onChange(enabled: boolean): void
}

export function InteriorCutawayToggle({ buildingId, title, enabled, onChange }: InteriorCutawayToggleProps) {
  return (
    <div className="space-y-1">
      <button
        aria-pressed={Boolean(buildingId && enabled)}
        className="rounded border px-3 py-2 text-sm disabled:opacity-50"
        disabled={!buildingId}
        onClick={() => onChange(!enabled)}
        type="button"
      >
        {enabled && buildingId ? 'Show exterior' : 'See inside'}{title && buildingId ? ` · ${title}` : ''}
      </button>
      <p className="text-xs text-muted-foreground">
        {buildingId ? 'Prototype interior · authored layout; shell fit pending review.' : 'Select a building to preview its interior.'}
      </p>
    </div>
  )
}

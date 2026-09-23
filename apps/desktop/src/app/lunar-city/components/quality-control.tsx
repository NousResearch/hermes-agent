import type { QualityTier } from '../model'

export type RendererStatus = 'ready' | 'degraded' | 'unavailable'

export interface QualityControlProps {
  tier?: QualityTier
  onTierChange?: (tier: QualityTier) => void
  rendererStatus?: RendererStatus
  disabled?: boolean
}

const QUALITY_OPTIONS: readonly { label: string; value: QualityTier }[] = [
  { label: 'Efficient (lowest GPU use)', value: 'efficient' },
  { label: 'Balanced', value: 'balanced' },
  { label: 'Detailed (highest GPU use)', value: 'detailed' }
]

const STATUS_COPY: Readonly<Record<RendererStatus, string>> = {
  degraded: 'Renderer degraded; selected quality remains truthful.',
  ready: 'Renderer ready.',
  unavailable: '3D renderer unavailable; accessible controls remain available.'
}

function tierLabel(tier: QualityTier): string {
  return tier.charAt(0).toUpperCase() + tier.slice(1)
}

export function QualityControl({
  disabled = false,
  onTierChange,
  rendererStatus = 'ready',
  tier = 'efficient'
}: QualityControlProps) {
  const status = `3D quality: ${tierLabel(tier)}. ${STATUS_COPY[rendererStatus]}`

  return (
    <section aria-label="3D quality control" className="lunar-city-quality-control">
      <label htmlFor="lunar-city-quality">3D quality</label>
      <select
        aria-describedby="lunar-city-quality-status"
        disabled={disabled}
        id="lunar-city-quality"
        onChange={event => onTierChange?.(event.target.value as QualityTier)}
        value={tier}
      >
        {QUALITY_OPTIONS.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <output aria-atomic="true" aria-live="polite" id="lunar-city-quality-status" role="status">
        {status}
      </output>
    </section>
  )
}

export { tierLabel }

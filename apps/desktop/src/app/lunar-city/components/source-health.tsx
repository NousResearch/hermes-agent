import type { AuthorityState, SourceHealth } from '../model'

export interface SourceHealthProps {
  sources: readonly SourceHealth[]
  formatTimestamp?: (observedAt: number) => string
}

const AUTHORITY_COPY: Readonly<Record<AuthorityState, { cue: string; label: string }>> = {
  authoritative: { cue: 'Healthy', label: 'Authoritative' },
  partial: { cue: 'Degraded', label: 'Partial' },
  stale: { cue: 'Stale data', label: 'Stale' },
  unknown: { cue: 'Unavailable', label: 'Unknown' }
}

function defaultTimestamp(observedAt: number): string {
  if (!Number.isFinite(observedAt) || observedAt <= 0) {
    return 'Unknown time'
  }

  return new Date(observedAt).toLocaleString()
}

/** A text-first source status surface; CSS color is intentionally optional. */
export function SourceHealthPanel({ formatTimestamp = defaultTimestamp, sources }: SourceHealthProps) {
  return (
    <section aria-label="Source health" aria-live="polite" className="lunar-city-source-health">
      <h2>Source health</h2>
      {sources.length === 0 ? (
        <p role="status">No source health data available.</p>
      ) : (
        <ul>
          {sources.map(source => {
            const copy = AUTHORITY_COPY[source.authority]

            return (
              <li key={`${source.source}:${source.observedAt}`}>
                <strong>{source.source}</strong>
                <span>
                  Status: {copy.label}; {copy.cue}
                </span>
                <span>Last observed: {formatTimestamp(source.observedAt)}</span>
                <span>Status cue: {copy.cue}</span>
                <span>{source.error ? `Error: ${source.error}` : 'No error reported'}</span>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

export { defaultTimestamp }

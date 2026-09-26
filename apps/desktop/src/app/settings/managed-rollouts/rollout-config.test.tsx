import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import type { RolloutCapabilities } from '@/lib/managed-rollout-contract'

import type { RolloutDraft } from './rollout-config'
import { canonicalWaves, observedDurations, WavePreview } from './wave-preview'

const draft: RolloutDraft = { mode: 'manual', concurrency: 1, canaryInstallId: 'b', selectedInstallIds: ['a', 'b', 'c'] }

const capabilities: RolloutCapabilities = {
  protocol: 1,
  available: true,
  reason: null,
  maxConcurrency: 1,
  maxInstallations: 10
}

function attempt(installId: string, startedAt: string | null, finishedAt: string | null) {
  return {
    identity: {
      connectionId: `ssh-${installId}`,
      installId,
      aliasConnectionIds: [],
      label: installId,
      displayAddress: `ssh-${installId}`,
      installationFingerprint: 'b'.repeat(64),
      sourceFingerprint: 'c'.repeat(64),
      admittedSha: 'a'.repeat(40)
    },
    correlationId: `correlation-${installId}`,
    wave: 0,
    phase: 'updated' as const,
    launchState: 'observed' as const,
    requiredScopeIds: null,
    skipReason: null,
    reprobes: 0,
    receipt: startedAt && finishedAt
      ? {
          correlationId: `correlation-${installId}`,
          installId,
          requestedSha: 'a'.repeat(40),
          preSha: null,
          postSha: 'a'.repeat(40),
          outcome: 'success',
          startedAt,
          finishedAt,
          stopReason: null
        }
      : null,
    health: null,
    recoveryRequired: false,
    reasons: []
  }
}

describe('managed rollout canonical draft', () => {
  it('keeps the stable canary first and excludes it from successor rows', () => {
    expect(canonicalWaves({ mode: 'manual', concurrency: 1, canaryInstallId: 'b', selectedInstallIds: ['a', 'b', 'c'] })).toEqual([['b'], ['a', 'c']])
  })

  it('measures only installations whose receipts recorded both ends of the run', () => {
    const durations = observedDurations([
      attempt('a', '2026-09-24T10:00:00.000Z', '2026-09-24T10:00:30.000Z'),
      attempt('b', '2026-09-24T10:00:00.000Z', null),
      { ...attempt('c', null, null), receipt: null }
    ])

    expect(durations).toEqual({ a: 30_000 })
  })

  it('renders the estimate for exactly the canonical wave plan it displays', () => {
    render(<I18nProvider configClient={null} initialLocale="en">
      <WavePreview capabilities={capabilities} draft={draft} durations={{ a: 30_000, b: 10_000, c: 20_000 }} planner={canonicalWaves} />
    </I18nProvider>)

    expect(screen.getByText('Wave 1: b')).toBeTruthy()
    expect(screen.getByText('Wave 2: a, c')).toBeTruthy()
    expect(screen.getByText(/Estimated machine time: 1m 00s/)).toBeTruthy()
  })

  it('shows an explicitly unavailable estimate instead of inventing timing', () => {
    render(<I18nProvider configClient={null} initialLocale="en">
      <WavePreview capabilities={capabilities} draft={draft} durations={{ a: 30_000 }} planner={canonicalWaves} />
    </I18nProvider>)

    expect(screen.getByText('Estimate unavailable.')).toBeTruthy()
    expect(screen.queryByText(/Estimated machine time/)).toBeNull()
  })

  it('refuses an estimate the advertised capability cannot execute', () => {
    render(<I18nProvider configClient={null} initialLocale="en">
      <WavePreview
        capabilities={{ ...capabilities, available: false, maxConcurrency: 0 }}
        draft={draft}
        durations={{ a: 30_000, b: 10_000, c: 20_000 }}
        planner={canonicalWaves}
      />
    </I18nProvider>)

    expect(screen.getByText('Estimate unavailable.')).toBeTruthy()
  })
})

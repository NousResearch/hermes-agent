// @vitest-environment jsdom
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { SourceHealth } from '../model'

import { SourceHealthPanel } from './source-health'

describe('SourceHealthPanel', () => {
  it('renders authority, last observed time, and diagnostics as text with non-color cues', () => {
    const sources: readonly SourceHealth[] = [
      { authority: 'authoritative', observedAt: 1_700_000_000_000, source: 'session:local' },
      {
        authority: 'partial',
        error: 'Kanban plugin unavailable',
        observedAt: 1_700_000_060_000,
        source: 'kanban:local'
      }
    ]

    render(<SourceHealthPanel formatTimestamp={timestamp => `T${timestamp}`} sources={sources} />)

    expect(screen.getByRole('region', { name: /source health/i })).toBeTruthy()
    expect(screen.getByText('session:local')).toBeTruthy()
    expect(screen.getByText(/Authoritative.*healthy/i)).toBeTruthy()
    expect(screen.getByText(/Last observed: T1700000000000/i)).toBeTruthy()
    expect(screen.getByText(/Partial.*degraded/i)).toBeTruthy()
    expect(screen.getByText(/Error: Kanban plugin unavailable/i)).toBeTruthy()
    expect(screen.getAllByText(/status cue:/i)).toHaveLength(2)
  })

  it('makes an empty source set explicit instead of implying that data is healthy', () => {
    render(<SourceHealthPanel sources={[]} />)

    expect(screen.getByText(/No source health data available/i)).toBeTruthy()
  })
})

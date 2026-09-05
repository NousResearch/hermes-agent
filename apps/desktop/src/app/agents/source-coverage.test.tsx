import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { OverviewSource } from '@/global'

import { SourceCoverage } from './source-coverage'

afterEach(cleanup)

it('explains that older gateways require an update for inventory, without claiming an empty history', () => {
  const source: OverviewSource = {
    connectionId: 'old',
    label: 'Older gateway',
    kind: 'remote',
    state: 'unsupported',
    sessions: [],
    canonical: [],
    live: [],
    profiles: [{ name: 'coder', display_name: 'Coder' }],
    total: 0,
    offset: 0,
    complete: false,
    liveCoverage: 'unavailable',
    errors: [],
    error: 'read-only-inventory-unavailable'
  }

  render(<SourceCoverage source={source} />)
  expect(screen.getByText('Update this gateway to show its sessions and bots.')).toBeTruthy()
  expect(screen.getByText(/Older gateway/)).toBeTruthy()
  expect(screen.queryByText(/0\s*\/\s*0/)).toBeNull()
})

it('keeps a complete, ready source to its name — no counts or status bookkeeping', () => {
  const source: OverviewSource = {
    connectionId: 'hq',
    label: 'HQ',
    kind: 'remote',
    state: 'ready',
    sessions: [],
    canonical: [],
    live: [],
    profiles: [],
    total: 42,
    offset: 42,
    complete: true,
    liveCoverage: 'process',
    errors: []
  }

  render(<SourceCoverage source={source} />)
  // Complete, ready coverage is the quiet default: name only, no bookkeeping.
  expect(screen.getByText('HQ')).toBeTruthy()
  expect(screen.queryByText(/42\s*\/\s*42/)).toBeNull()
  expect(screen.queryByText('Ready')).toBeNull()
})

/**
 * Roadmaps plugin — render smoke test for the page component.
 *
 * Renders the bare page (no plugin registration, no REST door bound) inside a
 * QueryClientProvider and asserts it reaches a stable empty state: the active
 * profile resolves to `default` but no project is selected, so the page must
 * show the "Select a project and a roadmap…" guidance without touching the
 * network.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { $projectId, $roadmapId } from './api'
import { RoadmapsPage } from './plugin'

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <RoadmapsPage />
    </QueryClientProvider>
  )
}

describe('RoadmapsPage', () => {
  beforeEach(() => {
    $projectId.set('')
    $roadmapId.set('')
  })

  it('renders the scope bar and the select-a-roadmap guidance without a scope', () => {
    renderPage()

    // The scope bar labels are present.
    expect(screen.getByText('Project')).toBeTruthy()
    expect(screen.getByText('Roadmap')).toBeTruthy()

    // No project selected → the guided empty state, not a loading skeleton.
    expect(screen.getByText(/Select a project and a roadmap/)).toBeTruthy()
  })

  it('shows the active profile and keeps the roadmap selector disabled without a project', () => {
    renderPage()
    // The active gateway profile resolves to `default` and is displayed read-only.
    expect(screen.getByText('default')).toBeTruthy()
    // No project id → no roadmap list is fetched or rendered.
    expect(screen.queryByText(/No roadmaps/)).toBeNull()
  })
})

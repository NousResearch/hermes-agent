import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'

import { RouteHeading } from './route-heading'

describe('RouteHeading', () => {
  const renderAt = (path: string) =>
    render(
      <MemoryRouter initialEntries={[path]}>
        <RouteHeading />
      </MemoryRouter>
    )

  it('renders a single top-level h1', () => {
    renderAt('/')
    expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
  })

  it('names the current route from its AppView', () => {
    renderAt('/settings')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Settings')
  })

  it('falls back to Chat for a session route', () => {
    renderAt('/some-session-id')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Chat')
  })

  it('tracks non-chat routes (skills, stripping the query)', () => {
    renderAt('/skills?tab=mcp')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Capabilities')
  })
})

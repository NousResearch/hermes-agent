import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { RouteAdvisoryBanner, shouldDisplayRouteAdvisory } from './route-advisory-banner'

describe('RouteAdvisoryBanner', () => {
  afterEach(() => cleanup())

  it('renders an advisory-only route recommendation without implying auto-switching', () => {
    render(
      <RouteAdvisoryBanner
        advisory={{
          advisory_mode: true,
          auto_execute: false,
          blocked_actions: ['raw_client_upload'],
          confidence: 8,
          profile: 'business-growth',
          requires_approval: true,
          route_id: 'business-growth'
        }}
      />
    )

    expect(screen.getByRole('status', { name: 'Route advisory' })).toBeTruthy()
    expect(screen.getByText(/Recommended profile: business-growth/i)).toBeTruthy()
    expect(screen.getByText(/confidence 8/i)).toBeTruthy()
    expect(screen.getByText(/Advisory only/i)).toBeTruthy()
    expect(screen.getByText(/No profile switch or auto-execution/i)).toBeTruthy()
    expect(screen.getByText(/Blocked actions: raw_client_upload/i)).toBeTruthy()
  })

  it('hides macos/main-hermes default decisions so normal prompts are quiet', () => {
    expect(
      shouldDisplayRouteAdvisory({
        advisory_mode: true,
        auto_execute: false,
        confidence: 0,
        profile: 'macos',
        route_id: 'main-hermes'
      })
    ).toBe(false)
  })
})

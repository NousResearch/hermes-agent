import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CapabilitiesHeader, type CapabilityHeaderMode } from './capabilities-header'
import type { CapabilityScope } from './scope-selector'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: { nav: { capabilities: 'Capabilities' } },
      settings: { profileScope: { appliesTo: 'Applies to' } },
      skills: {
        tabSkills: 'Skills',
        tabToolsets: 'Tools',
        tabPlugins: 'Plugins',
        hub: { landingHint: 'Extend Hermes with reusable skills.' },
        changesApplyNewSessions: 'Tool changes apply to new sessions.',
        plugins: { pageBlurb: 'Extend Hermes with plugins.' }
      },
      connectorsPage: { title: 'Connectors' },
      connectors: { disclaimer: 'Connect external services.' }
    }
  })
}))

afterEach(cleanup)

const labels: Record<CapabilityHeaderMode, string> = {
  skills: 'Skills',
  toolsets: 'Tools',
  connectors: 'Connectors',
  plugins: 'Plugins'
}

function makeScope(onChange = vi.fn()): CapabilityScope {
  return {
    crossBackend: false,
    key: 'local::default',
    label: 'default',
    onChange,
    options: [
      { key: 'local::default', label: 'default', value: 'default' },
      { key: 'local::researcher', label: 'researcher', value: 'researcher' }
    ],
    profile: 'default',
    value: 'default'
  } as CapabilityScope
}

describe('CapabilitiesHeader contract', () => {
  for (const mode of Object.keys(labels) as CapabilityHeaderMode[]) {
    it(`keeps breadcrumb and heading consistent for ${mode}`, () => {
      render(<CapabilitiesHeader mode={mode} scope={makeScope()} />)

      expect(screen.getByLabelText('Capabilities').textContent).toContain(labels[mode])
      expect(screen.getByRole('heading', { level: 1, name: labels[mode] })).toBeTruthy()
      expect(screen.getByRole('group', { name: 'Applies to' })).toBeTruthy()
    })
  }

  it('exposes profile scope as keyboard-reachable buttons with one selected value', () => {
    const onChange = vi.fn()
    render(<CapabilitiesHeader mode="skills" scope={makeScope(onChange)} />)

    const defaultProfile = screen.getByRole('button', { name: 'default' })
    const researcher = screen.getByRole('button', { name: 'researcher' })

    expect(defaultProfile.getAttribute('aria-pressed')).toBe('true')
    expect(researcher.getAttribute('aria-pressed')).toBe('false')

    researcher.focus()
    expect(document.activeElement).toBe(researcher)

    fireEvent.click(researcher)
    expect(onChange).toHaveBeenCalledWith('researcher')
  })

  it('hides Applies to when there is no scope choice', () => {
    const scope = makeScope()
    scope.options = [scope.options[0]]

    render(<CapabilitiesHeader mode="plugins" scope={scope} />)

    expect(screen.queryByRole('group', { name: 'Applies to' })).toBeNull()
  })
})

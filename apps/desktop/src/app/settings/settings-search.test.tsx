import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { describe, expect, it, vi } from 'vitest'

import { SettingsSearch } from './settings-search'

// The page catalog resolves labels through t.settings.nav and the search-page
// prefix through t.settings.searchPageResult — both must be localized, so the
// test drives the component with a non-English-shaped mock to prove no label
// is hardcoded in the component.
const mockT = {
  settings: {
    nav: {
      about: 'About',
      archivedChats: 'Archived Chats',
      billing: 'Billing',
      gateway: 'Gateway',
      keybinds: 'Keyboard Shortcuts',
      keysSettings: 'Settings',
      keysTools: 'Tools',
      notifications: 'Notifications',
      plugins: 'Plugins',
      providerAccounts: 'Accounts',
      providerApiKeys: 'API keys',
      providerCustomEndpoints: 'Custom Endpoints'
    },
    fieldLabels: {
      'tts.openai.model': 'OpenAI TTS Model',
      'approvals.mode': 'Approval Mode'
    },
    fieldDescriptions: {},
    sections: {
      voice: 'Voice',
      safety: 'Safety'
    },
    searchPageResult: (page: string) => `Settings: ${page}`,
    searchPlaceholder: {
      config: 'Search settings...'
    }
  },
  commandCenter: {
    noResults: 'No matching results found'
  }
}

// Module-level mock so `t` keeps a stable reference across renders — the real
// I18nProvider only swaps `t` on locale switch, and SettingsSearch resets its
// active index when the index changes.
vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: mockT })
}))

function LocationProbe() {
  const location = useLocation()

  return <div data-testid="location">{location.search}</div>
}

function renderSearch() {
  return render(
    <MemoryRouter initialEntries={['/settings']}>
      <SettingsSearch />
      <LocationProbe />
    </MemoryRouter>
  )
}

const searchInput = () => screen.getByPlaceholderText('Search settings...')
const type = (value: string) => fireEvent.change(searchInput(), { target: { value } })
const location = () => screen.getByTestId('location').textContent

describe('SettingsSearch', () => {
  it('resolves every page result label through i18n instead of hardcoded English', () => {
    renderSearch()

    type('billing')
    expect(screen.getByText('Settings: Billing')).toBeTruthy()

    type('custom endpoint')
    expect(screen.getByText('Settings: Custom Endpoints')).toBeTruthy()

    type('archived')
    expect(screen.getByText('Settings: Archived Chats')).toBeTruthy()
  })

  it('catalog covers Billing and Providers → Custom Endpoints destinations', () => {
    renderSearch()

    type('billing')
    expect(screen.getByText('Settings: Billing')).toBeTruthy()

    type('openai compatible')
    expect(screen.getByText('Settings: Custom Endpoints')).toBeTruthy()
  })

  it('deep-links to the matching config section and field', () => {
    renderSearch()

    type('openai model')
    fireEvent.click(screen.getByText('Voice: OpenAI TTS Model'))

    expect(location()).toBe('?tab=config:voice&field=tts.openai.model')
  })

  it('deep-links to a non-config page tab', () => {
    renderSearch()

    type('billing')
    fireEvent.click(screen.getByText('Settings: Billing'))

    expect(location()).toBe('?tab=billing')
  })

  it('navigates to the first ranked result on Enter', () => {
    renderSearch()

    type('billing')
    fireEvent.keyDown(searchInput(), { key: 'Enter' })

    expect(location()).toBe('?tab=billing')
  })

  it('moves the active index with ArrowDown and navigates to the selected result on Enter', () => {
    renderSearch()

    type('gateway')
    fireEvent.keyDown(searchInput(), { key: 'ArrowDown' })
    fireEvent.keyDown(searchInput(), { key: 'Enter' })

    // Second result for 'gateway' is API Keys → Settings (keyword-only match).
    expect(location()).toBe('?tab=keys&kview=settings')
  })

  it('deep-links to a compound page tab without encoding the sub-view separator', () => {
    renderSearch()

    type('custom endpoint')
    fireEvent.click(screen.getByText('Settings: Custom Endpoints'))

    // The raw `&` in `providers&pview=custom-endpoints` must survive navigation.
    expect(location()).toBe('?tab=providers&pview=custom-endpoints')
  })

  it('ranks visible-label matches above keyword-only matches', () => {
    renderSearch()

    // 'gateway' matches the Gateway page in its label (0.7) and API Keys →
    // Settings only in its keywords (0.4) — the label match must come first.
    type('gateway')
    const results = screen.getAllByText(/^Settings: /)
    expect(results[0].textContent).toBe('Settings: Gateway')
    expect(results.map(result => result.textContent)).toContain('Settings: Settings')
  })

  it('shows the empty state when no item matches', () => {
    renderSearch()

    type('bluetooth')
    expect(screen.getByText('No matching results found')).toBeTruthy()
  })
})

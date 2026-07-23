import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { configSettingsRender, modelSettingsRender } = vi.hoisted(() => ({
  configSettingsRender: vi.fn(),
  modelSettingsRender: vi.fn()
}))

vi.mock('./about-settings', () => ({ AboutSettings: () => null }))
vi.mock('./appearance-settings', () => ({ AppearanceSettings: () => null }))
vi.mock('./billing', () => ({ BillingSettings: () => null }))
vi.mock('./gateway-settings', () => ({ GatewaySettings: () => null }))
vi.mock('./keybind-settings', () => ({ KeybindSettings: () => null }))
vi.mock('./keys-settings', () => ({
  KEYS_VIEWS: ['tools', 'settings'],
  KeysSettings: () => null
}))
vi.mock('./notifications-settings', () => ({ NotificationsSettings: () => null }))
vi.mock('./plugins-settings', () => ({ PluginsSettings: () => null }))
vi.mock('./providers-settings', () => ({
  PROVIDER_VIEWS: ['accounts', 'keys'],
  ProvidersSettings: () => null
}))
vi.mock('./sessions-settings', () => ({ SessionsSettings: () => null }))
vi.mock('./config-settings', () => ({
  ConfigSettings: (props: { activeSectionId: string; onUseMoaPreset?: (name: string) => void }) => {
    configSettingsRender(props)

    return (
      <div data-testid="config-settings">
        <span>{props.activeSectionId}</span>
        <button onClick={() => props.onUseMoaPreset?.('embedded-preset')} type="button">
          Use embedded preset
        </button>
      </div>
    )
  }
}))
vi.mock('./model-settings', () => ({
  ModelSettings: (props: { onUseMoaPreset?: (name: string) => void; studioOnly?: boolean }) => {
    modelSettingsRender(props)

    return (
      <div data-testid="model-settings">
        <span>{props.studioOnly ? 'studio-only' : 'embedded'}</span>
        <button onClick={() => props.onUseMoaPreset?.('studio-preset')} type="button">
          Use studio preset
        </button>
      </div>
    )
  }
}))

import { SettingsView } from './index'

function LocationProbe() {
  const location = useLocation()

  return <output data-testid="location">{`${location.pathname}${location.search}`}</output>
}

function renderSettings(initialEntry: string, onUseMoaPreset = vi.fn()) {
  return {
    onUseMoaPreset,
    view: render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <SettingsView onClose={vi.fn()} onUseMoaPreset={onUseMoaPreset} />
        <LocationProbe />
      </MemoryRouter>
    )
  }
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Settings MoA Studio route', () => {
  it('renders the dedicated studio-only view from a direct ?tab=moa deep link', () => {
    const { onUseMoaPreset } = renderSettings('/settings?tab=moa')

    expect(screen.getByTestId('model-settings').textContent).toContain('studio-only')
    expect(screen.queryByTestId('config-settings')).toBeNull()
    expect(modelSettingsRender).toHaveBeenLastCalledWith(
      expect.objectContaining({ onUseMoaPreset, studioOnly: true })
    )

    fireEvent.click(screen.getByRole('button', { name: 'Use studio preset' }))
    expect(onUseMoaPreset).toHaveBeenCalledWith('studio-preset')
  })

  it('navigates to MoA Studio from the settings rail and keeps the embedded Model editor on the shared callback', () => {
    const { onUseMoaPreset } = renderSettings('/settings')

    expect(screen.getByTestId('config-settings').textContent).toContain('model')
    fireEvent.click(screen.getByRole('button', { name: 'MoA Studio' }))

    expect(screen.getByTestId('location').textContent).toBe('/settings?tab=moa')
    expect(screen.getByTestId('model-settings').textContent).toContain('studio-only')

    fireEvent.click(screen.getAllByRole('button', { name: 'Model' })[0])
    expect(screen.getByTestId('config-settings').textContent).toContain('model')
    expect(configSettingsRender).toHaveBeenLastCalledWith(
      expect.objectContaining({ activeSectionId: 'model', onUseMoaPreset })
    )

    fireEvent.click(screen.getByRole('button', { name: 'Use embedded preset' }))
    expect(onUseMoaPreset).toHaveBeenCalledWith('embedded-preset')
  })
})

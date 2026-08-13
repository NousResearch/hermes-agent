import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { ModelOptionProvider } from '@/types/hermes'

import { ProviderConfigurePanel } from './provider-configure-panel'

const provider: ModelOptionProvider = {
  slug: 'deepseek',
  name: 'DeepSeek',
  models: [],
  auth_type: 'api_key',
  key_env: 'DEEPSEEK_API_KEY',
  authenticated: false
}

const renderPanel = (over: Partial<React.ComponentProps<typeof ProviderConfigurePanel>> = {}) =>
  render(
    <ProviderConfigurePanel
      error={null}
      onConfigure={over.onConfigure ?? vi.fn()}
      provider={provider}
      working={over.working ?? false}
      {...over}
    />
  )

describe('ProviderConfigurePanel', () => {
  it('renders the provider name and description', () => {
    renderPanel()

    expect(screen.getByText('DeepSeek')).toBeTruthy()
    expect(screen.getByText(/Enter your API key/i)).toBeTruthy()
  })

  it('disables the Save button when the API key is empty', () => {
    renderPanel()

    const button = screen.getByRole('button', { name: /Save & discover models/i }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
  })

  it('enables the Save button once an API key is entered', () => {
    renderPanel()

    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sk-123' } })

    const button = screen.getByRole('button', { name: /Save & discover models/i }) as HTMLButtonElement
    expect(button.disabled).toBe(false)
  })

  it('calls onConfigure with the key and undefined base URL when base URL is empty', () => {
    const onConfigure = vi.fn()
    renderPanel({ onConfigure })

    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sk-123' } })
    fireEvent.click(screen.getByRole('button', { name: /Save & discover models/i }))

    expect(onConfigure).toHaveBeenCalledWith('sk-123', undefined)
  })

  it('calls onConfigure with the key and base URL when a base URL is provided', () => {
    const onConfigure = vi.fn()
    renderPanel({ onConfigure })

    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sk-123' } })
    fireEvent.change(screen.getByLabelText('Base URL override'), { target: { value: 'https://my.host/v1' } })
    fireEvent.click(screen.getByRole('button', { name: /Save & discover models/i }))

    expect(onConfigure).toHaveBeenCalledWith('sk-123', 'https://my.host/v1')
  })

  it('blocks submit and shows an error for an invalid base URL', () => {
    const onConfigure = vi.fn()
    renderPanel({ onConfigure })

    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sk-123' } })
    fireEvent.change(screen.getByLabelText('Base URL override'), { target: { value: 'not-a-url' } })

    const button = screen.getByRole('button', { name: /Save & discover models/i }) as HTMLButtonElement
    expect(button.disabled).toBe(true)

    fireEvent.click(button)
    expect(onConfigure).not.toHaveBeenCalled()
    expect(screen.getByText('Base URL must be a valid http(s) URL.')).toBeTruthy()
  })

  it('shows the error banner when the error prop is set', () => {
    renderPanel({ error: 'Could not reach the provider.' })

    expect(screen.getByText('Could not reach the provider.')).toBeTruthy()
  })

  it('disables inputs and the button while working', () => {
    renderPanel({ working: true })

    expect((screen.getByLabelText('API key') as HTMLInputElement).disabled).toBe(true)
    expect((screen.getByLabelText('Base URL override') as HTMLInputElement).disabled).toBe(true)
    const button = screen.getByRole('button', { name: /Saving/i }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
  })
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PluginSettingField } from '@/store/agent-plugins'

import { collectChanges, initialDraft, PluginSettingsForm } from './plugin-settings-form'

const FIELDS: PluginSettingField[] = [
  {
    description: 'Service endpoint',
    key: 'api_url',
    label: 'API URL',
    required: true,
    type: 'string',
    value: 'https://a'
  },
  { description: '', key: 'retries', label: 'Retries', required: false, type: 'number', value: 3 },
  { description: '', key: 'verbose', label: 'Verbose', required: false, type: 'boolean', value: false },
  {
    choices: ['fast', 'careful'],
    description: '',
    key: 'mode',
    label: 'Mode',
    required: false,
    type: 'enum',
    value: 'fast'
  },
  {
    description: 'Token',
    env: 'DEMO_API_KEY',
    has_value: true,
    key: 'api_key',
    label: 'API key',
    required: false,
    type: 'secret'
  },
  { description: '', key: 'extra', label: 'Extra', required: false, type: 'json', value: { a: 1 } }
]

describe('PluginSettingsForm (#46600, #87934)', () => {
  afterEach(cleanup)

  it('renders one control per schema type from the table, secrets masked with no value echoed', () => {
    render(<PluginSettingsForm disabled={false} fields={FIELDS} idPrefix="p" onSave={vi.fn(async () => true)} />)

    expect((screen.getByLabelText('API URL') as HTMLInputElement).value).toBe('https://a')
    expect((screen.getByLabelText(/^Retries/) as HTMLInputElement).type).toBe('number')
    expect(screen.getByRole('switch', { name: 'Verbose' }).getAttribute('aria-checked')).toBe('false')
    expect(screen.getByRole('combobox', { name: 'Mode' })).toBeTruthy()
    const secret = screen.getByLabelText(/^API key/) as HTMLInputElement
    expect(secret.type).toBe('password')
    expect(secret.value).toBe('')
    expect(secret.placeholder).toContain('set')
    expect((screen.getByLabelText(/^Extra/) as HTMLTextAreaElement).value).toBe(JSON.stringify({ a: 1 }, null, 2))
    // Nothing changed yet → nothing to save.
    expect((screen.getByRole('button', { name: 'Save settings' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('submits only what changed, coerced to wire types, secrets routed by env name; clears secrets after save', async () => {
    const onSave = vi.fn(async () => true)
    render(<PluginSettingsForm disabled={false} fields={FIELDS} idPrefix="p" onSave={onSave} />)

    fireEvent.change(screen.getByLabelText(/^Retries/), { target: { value: '9' } })
    fireEvent.click(screen.getByRole('switch', { name: 'Verbose' }))
    fireEvent.change(screen.getByLabelText(/^API key/), { target: { value: 'sk-x' } })
    fireEvent.submit(screen.getByTestId('p-settings-form'))

    await vi.waitFor(() =>
      expect(onSave).toHaveBeenCalledWith({ secrets: { DEMO_API_KEY: 'sk-x' }, values: { retries: 9, verbose: true } })
    )
    await vi.waitFor(() => expect((screen.getByLabelText(/^API key/) as HTMLInputElement).value).toBe(''))

    // A value the plugin cannot accept never reaches the backend.
    expect(() => collectChanges(FIELDS, { ...initialDraft(FIELDS), mode: 'reckless' })).toThrow(/Mode/)
    expect(() => collectChanges(FIELDS, { ...initialDraft(FIELDS), retries: 'five' })).toThrow(/number/)
  })
})

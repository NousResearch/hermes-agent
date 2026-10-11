import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PluginSettingField } from '@/store/agent-plugins'

import {
  collectChanges,
  fieldLabel,
  groupFieldsBySection,
  humanizeSettingKey,
  initialDraft,
  joinSentences,
  PluginSettingsForm
} from './plugin-settings-form'

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
    render(
      <PluginSettingsForm disabled={false} fields={FIELDS} idPrefix="p" onSave={vi.fn(async () => true)} title="Demo" />
    )

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
    render(<PluginSettingsForm disabled={false} fields={FIELDS} idPrefix="p" onSave={onSave} title="Demo" />)

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

  it('labels raw keys like native rows and joins helper sentences with punctuation', () => {
    const fields: PluginSettingField[] = [
      {
        description: 'Per-day budget',
        key: 'daily_budget',
        label: 'daily_budget',
        required: false,
        type: 'number',
        value: 1
      },
      {
        description: 'Optional maps key for travel-time estimates',
        env: 'TRIP_NOTES_MAPS_API_KEY',
        has_value: false,
        key: 'maps_api_key',
        label: 'maps_api_key',
        required: true,
        type: 'secret'
      }
    ]

    render(
      <PluginSettingsForm disabled={false} fields={fields} idPrefix="t" onSave={vi.fn(async () => true)} title="Trip" />
    )

    expect(screen.getByLabelText('Daily budget')).toBeTruthy()
    expect(screen.getByLabelText('Maps API key')).toBeTruthy()
    expect(screen.queryByText('daily_budget')).toBeNull()
    expect(
      screen.getByText(
        "Optional maps key for travel-time estimates. Stored in the profile's .env as TRIP_NOTES_MAPS_API_KEY, never in config.yaml; leave blank to keep the current value."
      )
    ).toBeTruthy()
    // Only required fields are marked; optional is the unmarked default, as in native Settings.
    expect(screen.getAllByText('Required')).toHaveLength(1)
    expect(screen.queryByText('(optional)')).toBeNull()
  })
})

describe('plugin settings labels and helper copy', () => {
  it('humanizes keys in sentence case, keeping acronyms upper-case', () => {
    expect(humanizeSettingKey('daily_budget')).toBe('Daily budget')
    expect(humanizeSettingKey('remind_me')).toBe('Remind me')
    expect(humanizeSettingKey('maps_api_key')).toBe('Maps API key')
    expect(humanizeSettingKey('webhook_url')).toBe('Webhook URL')
    expect(humanizeSettingKey('mcp_server_id')).toBe('MCP server ID')
    expect(humanizeSettingKey('apiBaseURL')).toBe('API base URL')
    expect(humanizeSettingKey('allowed-channel-ids')).toBe('Allowed channel IDs')
    expect(humanizeSettingKey('units')).toBe('Units')
  })

  it("prefers the manifest's label/title over the humanized key", () => {
    expect(fieldLabel({ key: 'units', label: 'Unit system' })).toBe('Unit system')
    // The backend falls back to the bare key when the manifest gave no label.
    expect(fieldLabel({ key: 'maps_api_key', label: 'maps_api_key' })).toBe('Maps API key')
  })

  it('ends each sentence before the next begins and drops blanks', () => {
    expect(joinSentences('Optional maps key', 'Stored in .env.')).toBe('Optional maps key. Stored in .env.')
    expect(joinSentences('Already ends.', 'Next')).toBe('Already ends. Next')
    expect(joinSentences('Ask first?', 'Then save.')).toBe('Ask first? Then save.')
    expect(joinSentences('', '  Only this  ', undefined)).toBe('Only this')
    expect(joinSentences('Lone description')).toBe('Lone description')
  })
})

describe('plugin settings section grouping', () => {
  const field = (key: string, section?: string): PluginSettingField => ({
    description: '',
    key,
    label: key,
    required: false,
    section,
    type: 'boolean',
    value: false
  })

  it('renders a manifest with no sections as ONE flat group', () => {
    const groups = groupFieldsBySection([field('a'), field('b'), field('c')])

    expect(groups).toHaveLength(1)
    expect(groups[0].section).toBeUndefined()
    expect(groups[0].fields.map(f => f.key)).toEqual(['a', 'b', 'c'])
  })

  it('collects each run of same-section fields under one heading', () => {
    const groups = groupFieldsBySection([
      field('enabled', 'Behaviour'),
      field('mode', 'Behaviour'),
      field('bar_width', 'Appearance'),
      field('emoji', 'Appearance')
    ])

    expect(groups.map(g => g.section)).toEqual(['Behaviour', 'Appearance'])
    expect(groups[0].fields.map(f => f.key)).toEqual(['enabled', 'mode'])
    expect(groups[1].fields.map(f => f.key)).toEqual(['bar_width', 'emoji'])
  })

  it('starts a new group when the section changes back', () => {
    const groups = groupFieldsBySection([
      field('a', 'One'),
      field('b', 'Two'),
      field('c', 'One')
    ])

    expect(groups.map(g => g.section)).toEqual(['One', 'Two', 'One'])
  })

  it('keeps ungrouped fields together in their own trailing group', () => {
    const groups = groupFieldsBySection([field('a', 'X'), field('b'), field('c')])

    expect(groups.map(g => g.section)).toEqual(['X', undefined])
    expect(groups[1].fields.map(f => f.key)).toEqual(['b', 'c'])
  })

  it('treats a blank section as ungrouped rather than an empty heading', () => {
    const groups = groupFieldsBySection([field('a', '   '), field('b')])

    expect(groups).toHaveLength(1)
    expect(groups[0].section).toBeUndefined()
  })

  it('preserves order and loses no field', () => {
    const fields = [field('a'), field('b', 'S'), field('c', 'S'), field('d'), field('e', 'T')]
    const flattened = groupFieldsBySection(fields).flatMap(g => g.fields)

    expect(flattened.map(f => f.key)).toEqual(['a', 'b', 'c', 'd', 'e'])
  })
})

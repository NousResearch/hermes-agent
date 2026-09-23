// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ConfigField } from './config-field'

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
})

describe('ConfigField option help', () => {
  it('keeps approval values stable while showing plain-language labels and help', async () => {
    const onChange = vi.fn()

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ConfigField
          enumOptions={['manual', 'smart', 'off']}
          onChange={onChange}
          optionDescriptions={{
            manual: 'Ask before actions that need approval',
            smart: 'Assess actions and ask only when needed',
            off: 'Approve actions without asking'
          }}
          optionLabels={{ manual: 'Ask', smart: 'Smart', off: 'Auto Approve' }}
          schema={{ type: 'select' }}
          schemaKey="approvals.mode"
          value="smart"
        />
      </I18nProvider>
    )

    fireEvent.click(screen.getByRole('combobox'))

    const ask = await screen.findByRole('option', {
      name: 'Ask',
      description: 'Ask before actions that need approval'
    })

    expect(
      screen.getByRole('option', {
        name: 'Smart',
        description: 'Assess actions and ask only when needed'
      })
    ).toBeTruthy()
    expect(
      screen.getByRole('option', {
        name: 'Auto Approve',
        description: 'Approve actions without asking'
      })
    ).toBeTruthy()
    expect(screen.getByText('Approve actions without asking')).toBeTruthy()

    fireEvent.click(ask)
    expect(onChange).toHaveBeenCalledWith('manual')
  })
})
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        config: { notSet: 'Not set' },
        fieldDescriptions: {},
        fieldLabels: {}
      }
    }
  })
}))

import { ConfigField } from './config-field'

afterEach(cleanup)

describe('ConfigField', () => {
  it('masks terminal backend secret fields', () => {
    const { container } = render(
      <ConfigField
        onChange={vi.fn()}
        schema={{ type: 'secret' }}
        schemaKey="terminal.backends.coder.token"
        value="sensitive-token"
      />
    )

    const input = container.querySelector('input')

    expect(input?.getAttribute('type')).toBe('password')
    expect(container.querySelector('textarea')).toBeNull()
  })
})

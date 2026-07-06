import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import DataModelRenderer from './data-model-embed'

afterEach(cleanup)

describe('DataModelRenderer', () => {
  it('renders entities and fields from valid JSON', () => {
    render(
      <DataModelRenderer
        code={JSON.stringify({
          title: 'Workstream schema',
          entities: [{ name: 'Workstream', fields: [{ name: 'sessionId', type: 'string', note: 'Desktop key' }] }]
        })}
      />
    )

    expect(screen.getByText('Workstream schema')).toBeTruthy()
    expect(screen.getByText('Workstream')).toBeTruthy()
    expect(screen.getByText('sessionId')).toBeTruthy()
    expect(screen.getByText('Desktop key')).toBeTruthy()
  })

  it('falls back on invalid JSON', () => {
    render(<DataModelRenderer code='{not-json' />)

    expect(screen.getByText('Invalid data-model block')).toBeTruthy()
  })

  it('renders attacker strings as text, not markup', () => {
    const { container } = render(
      <DataModelRenderer
        code={JSON.stringify({
          entities: [{ name: '<img src=x onerror=alert(1)>', fields: [{ name: '<script>alert(1)</script>', type: 'string' }] }]
        })}
      />
    )

    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(screen.getByText('<script>alert(1)</script>')).toBeTruthy()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
  })
})

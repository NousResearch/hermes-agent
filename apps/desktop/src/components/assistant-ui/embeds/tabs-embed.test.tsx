import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import TabsRenderer from './tabs-embed'

afterEach(cleanup)

describe('TabsRenderer', () => {
  it('renders tabs from valid JSON and switches content', () => {
    render(
      <TabsRenderer
        code={JSON.stringify({
          title: 'Approaches',
          tabs: [
            { title: 'Small', content: 'Minimal path' },
            { title: 'Large', content: 'Full rewrite' }
          ]
        })}
      />
    )

    expect(screen.getByText('Approaches')).toBeTruthy()
    expect(screen.getAllByText('Small').length).toBeGreaterThan(0)
    expect(screen.getByText('Minimal path')).toBeTruthy()

    fireEvent.click(screen.getByText('Large'))

    expect(screen.getByText('Full rewrite')).toBeTruthy()
  })

  it('falls back on invalid JSON', () => {
    render(<TabsRenderer code='{"tabs": []}' />)

    expect(screen.getByText('Invalid tabs block')).toBeTruthy()
  })

  it('renders attacker strings as text, not markup', () => {
    const { container } = render(
      <TabsRenderer code={JSON.stringify({ tabs: [{ title: '<script>alert(1)</script>', content: '<img src=x onerror=alert(1)>' }] })} />
    )

    expect(screen.getAllByText('<script>alert(1)</script>').length).toBeGreaterThan(0)
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })
})

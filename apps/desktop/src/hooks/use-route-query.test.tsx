import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it } from 'vitest'

import { useRouteQuery } from './use-route-query'

afterEach(cleanup)

function Probe({ path }: { path: string }) {
  const query = useRouteQuery(path)

  return <output aria-label="query">{query ? query.params.toString() : 'none'}</output>
}

const read = () => screen.getByRole('status', { name: 'query' }).textContent

describe('useRouteQuery', () => {
  it("reads the query only while the app is on the page's own path", () => {
    render(
      <MemoryRouter initialEntries={['/kanban?board=a&task=b']}>
        <Probe path="/kanban" />
      </MemoryRouter>
    )

    expect(read()).toBe('board=a&task=b')

    cleanup()
    render(
      <MemoryRouter initialEntries={['/settings?board=a']}>
        <Probe path="/kanban" />
      </MemoryRouter>
    )

    expect(read()).toBe('none')
  })

  it('is inert outside a router', () => {
    render(<Probe path="/kanban" />)

    expect(read()).toBe('none')
  })
})

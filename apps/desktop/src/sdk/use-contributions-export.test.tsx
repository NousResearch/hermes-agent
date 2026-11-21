import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { SIDEBAR_NAV_AREA, useContributions } from '@/sdk'

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

function NavIds() {
  const rows = useContributions(SIDEBAR_NAV_AREA)

  return <ul>{rows.map(c => <li key={c.id}>{c.id}</li>)}</ul>
}

const ids = () => screen.queryAllByRole('listitem').map(li => li.textContent)

describe('useContributions from the plugin SDK', () => {
  it('lists an area in order and follows registrations and disposals', () => {
    render(<NavIds />)
    const before = ids()

    act(() => {
      disposers.push(registry.register({ id: 'sdk-test:late', area: SIDEBAR_NAV_AREA, order: 60, data: {} }))
      disposers.push(registry.register({ id: 'sdk-test:early', area: SIDEBAR_NAV_AREA, order: 55, data: {} }))
    })

    expect(ids()).toEqual([...before, 'sdk-test:early', 'sdk-test:late'])

    act(() => disposers.shift()?.())

    expect(ids()).toEqual([...before, 'sdk-test:early'])
  })

  it('keeps the same array between renders while the area is unchanged', () => {
    disposers.push(registry.register({ id: 'sdk-test:row', area: SIDEBAR_NAV_AREA, order: 1, data: {} }))
    const seen: Array<readonly unknown[]> = []

    function Probe({ tick }: { tick: number }) {
      seen.push(useContributions(SIDEBAR_NAV_AREA))

      return <span>{tick}</span>
    }

    const { rerender } = render(<Probe tick={0} />)
    rerender(<Probe tick={1} />)
    expect(seen.length).toBeGreaterThanOrEqual(2)
    expect(seen.at(-1)).toBe(seen[0])
  })
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ConnectorsDirectory } from './connectors-directory'
import { deriveCards } from './derive'
import { HOSTED, LOCAL, TITLES } from './fixtures'
import type { ConnectorCardModel, ConnectorsFilter } from './types'

const ALL: ConnectorsFilter = { category: null, pill: 'all', query: '', residency: null }
const allCards = deriveCards({ hosted: HOSTED, local: LOCAL, titles: TITLES })

const onOpen = vi.fn()

/** The page owns its filter, so the test drives the real controlled contract
 *  rather than re-rendering with a hand-written filter each time. */
function Harness({ cards = allCards, hostedFailed = false }: { cards?: ConnectorCardModel[]; hostedFailed?: boolean }) {
  const [filter, setFilter] = useState<ConnectorsFilter>(ALL)

  return (
    <ConnectorsDirectory
      cards={cards}
      filter={filter}
      hostedFailed={hostedFailed}
      onFilterChange={setFilter}
      onOpen={onOpen}
      onRetryHosted={() => {}}
      onServerToggle={() => {}}
      onVerb={() => {}}
    />
  )
}

const pill = (name: RegExp) => screen.queryByRole('button', { name })

afterEach(cleanup)

describe('the state pills', () => {
  it('shows one pill per state that has something in it', () => {
    render(<Harness />)

    expect(pill(/^All 10$/)).toBeTruthy()
    expect(pill(/^Needs attention 3$/)).toBeTruthy()
    expect(pill(/^On this Mac 3$/)).toBeTruthy()
    expect(pill(/^Turned off 2$/)).toBeTruthy()
  })

  it('omits a pill with a zero count', () => {
    render(<Harness cards={allCards.filter(card => card.state !== 'off')} />)

    expect(pill(/^All 8$/)).toBeTruthy()
    expect(pill(/Turned off/)).toBeNull()
  })

  it('recounts the pills under the search, because a pill promises what it will show', () => {
    render(<Harness />)

    fireEvent.change(screen.getByRole('textbox', { name: /Search/ }), { target: { value: 'git' } })

    expect(pill(/^All 1$/)).toBeTruthy()
    expect(pill(/Turned off/)).toBeNull()
  })
})

describe('the groups', () => {
  it('orders them and drops the empty ones', () => {
    render(<Harness />)

    const headings = screen.getAllByRole('heading', { level: 3 }).map(node => node.textContent)

    expect(headings).toEqual(['Connected', 'On this Mac', 'Available', 'Turned off'])
  })

  it('keeps the servers on this Mac when only the hosted half failed', () => {
    render(<Harness cards={allCards.filter(card => card.residency === 'local')} hostedFailed />)

    expect(screen.getByText('Could not reach your Nous apps.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    expect(screen.getByRole('heading', { level: 3, name: 'On this Mac' })).toBeTruthy()
    expect(screen.getByText('https://api.githubcopilot.com/mcp/')).toBeTruthy()
  })

  it('keeps the account note verbatim', () => {
    render(<Harness />)

    expect(screen.getByText('Nous apps follow your account, not the profile.')).toBeTruthy()
  })
})

describe('the row cards', () => {
  it('opens the app from the name, in any group', () => {
    render(<Harness />)

    fireEvent.click(screen.getByRole('button', { name: /^Gmail/ }))

    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ slug: 'gmail' }))
  })

  it('gives a server on this Mac its switch, so the dialog is never needed to turn one off', () => {
    render(<Harness />)

    expect(screen.getByRole('switch', { name: 'Turn Postgres off' })).toBeTruthy()
  })

  it('shows one verb per broken app and none for one the organisation took away', () => {
    render(<Harness />)

    expect(screen.getByRole('button', { name: 'Reconnect' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
    expect(screen.getByText('Off by your organisation')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Turn back on' })).toBeTruthy()
  })

  it('lets a broken server be repaired, not only switched off', () => {
    render(<Harness />)

    // The switch and the verb share one lane, so a lane that could hold only one
    // of them left `Authenticate` on screen with no way to press it.
    expect(screen.getByRole('switch', { name: 'Turn Linear off' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Authenticate' })).toBeTruthy()
  })

  it('announces the catalog mark instead of hiding it from assistive tech', () => {
    render(<Harness />)

    expect(screen.getAllByRole('img', { name: 'In the Hermes catalog' }).length).toBeGreaterThan(0)
  })

  it('marks the one card that is showing a local backing over a hosted twin', () => {
    render(<Harness />)

    expect(screen.getByText('Hosted version available')).toBeTruthy()
  })
})

describe('when nothing matches', () => {
  it('keeps the search and names the way out', () => {
    render(<Harness />)

    fireEvent.change(screen.getByRole('textbox', { name: /Search/ }), { target: { value: 'quickbooks' } })

    expect(screen.getByText('No matching apps')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Clear the search' }))

    expect(screen.getByRole('heading', { level: 3, name: 'Connected' })).toBeTruthy()
  })

  it('reads a first run as a first run, not as a failure', () => {
    render(<Harness cards={[]} />)

    expect(screen.getByText('No apps in the catalog yet.')).toBeTruthy()
  })

  it('hides the search field when there is nothing to search', () => {
    render(<Harness cards={[]} />)

    expect(screen.queryByRole('textbox', { name: /Search/ })).toBeNull()
  })
})

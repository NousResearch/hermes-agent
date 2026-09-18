import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ConnectorDialog } from './connector-dialog'
import { deriveCards } from './derive'
import { HOSTED, LOCAL, TITLES } from './fixtures'

const cards = deriveCards({ hosted: HOSTED, local: LOCAL, titles: TITLES })
const cardFor = (slug: string) => cards.find(card => card.slug === slug)!

afterEach(cleanup)

describe('opening the dialog', () => {
  it('does not hand the keyboard a control that turns something off', async () => {
    render(
      <ConnectorDialog card={cardFor('postgres')} onOpenChange={() => {}} onServerToggle={() => {}} open tools={null} />
    )

    const title = screen.getByText('Postgres')

    // Radix focuses the first tabbable node, which on a local card is the server
    // switch and on a connected hosted one is Disconnect. The title takes it
    // instead: a dialog that opens on Enter must not open on "off".
    await waitFor(() => {
      expect(title.ownerDocument.activeElement).toBe(title)
    })
    expect(screen.getByRole('switch', { name: /Postgres/ })).not.toBe(title.ownerDocument.activeElement)
  })
})

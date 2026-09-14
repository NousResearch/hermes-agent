import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { StatusbarControls, type StatusbarItem } from '@/app/shell/statusbar-controls'
import { $statusbarHiddenIds, $statusbarVisible, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

const openExternalLink = vi.fn()

vi.mock('@/lib/external-link', () => ({
  openExternalLink: (href: string) => openExternalLink(href)
}))

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
  $statusbarVisible.set(true)
})

describe('statusbar external href', () => {
  it('opens a link-variant item href through the validated external opener', () => {
    const href = 'https://hermes-agent.nousresearch.com/docs/user-guide/desktop'

    const item: StatusbarItem = {
      href,
      id: 'docs-link',
      label: 'Docs',
      lockedVisible: true,
      variant: 'link'
    }

    render(
      <MemoryRouter>
        <StatusbarControls items={[item]} />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole('link', { name: 'Docs' }))

    expect(openExternalLink).toHaveBeenCalledWith(href)
  })
})

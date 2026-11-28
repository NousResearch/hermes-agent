import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, useNavigate } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { KeptChatRoutes, LayoutTreeCover } from './route-cover'

afterEach(cleanup)

let go: (to: string) => void = () => undefined

function Navigator() {
  go = useNavigate()

  return null
}

// The redesign's nesting: a route overlay covers the whole layout tree, and inside it the kept chat
// is covered by any workspace page.
function Shell() {
  return (
    <MemoryRouter initialEntries={['/s1']}>
      <Navigator />
      <LayoutTreeCover>
        <KeptChatRoutes
          chat={<Route element={<textarea aria-label="Composer" />} path="/:sessionId" />}
          pages={<Route element={<p>Library</p>} path="/artifacts" />}
        />
      </LayoutTreeCover>
    </MemoryRouter>
  )
}

const composer = () => screen.getByLabelText('Composer')

/** Opening a cover makes the chat inert, which blurs it in a browser (jsdom only marks it). */
function leaveChatFor(to: string) {
  act(() => go(to))
  act(() => composer().blur())
}

it('keeps the chat mounted, inert and hidden under an overlay and under a page', () => {
  render(<Shell />)
  const kept = composer()

  leaveChatFor('/settings')
  expect(kept.closest('[inert]')).not.toBeNull()

  act(() => go('/artifacts'))
  expect(screen.getByText('Library')).toBeTruthy()
  expect(composer()).toBe(kept)
  expect(kept.closest('[aria-hidden="true"]')).not.toBeNull()
  expect(kept.closest('[inert]')).not.toBeNull()
})

it('gives focus back to the chat only once the chat itself is uncovered', () => {
  render(<Shell />)
  act(() => composer().focus())

  leaveChatFor('/settings')
  // The overlay gives way to a page: the layout tree uncovers but the chat stays covered, so the
  // focus the tree remembers (the composer) must not be handed back into the still-inert chat.
  act(() => go('/artifacts'))
  expect(document.activeElement).toBe(document.body)

  act(() => go('/s1'))
  expect(document.activeElement).toBe(composer())
})

it('gives focus back to the chat when an overlay closes straight onto it', () => {
  render(<Shell />)
  act(() => composer().focus())

  leaveChatFor('/settings')
  act(() => go('/s1'))
  expect(document.activeElement).toBe(composer())
})

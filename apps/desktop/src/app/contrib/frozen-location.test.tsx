import { act, cleanup, render, screen } from '@testing-library/react'
import { type ReactNode, useState } from 'react'
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { AtLocation, type RouterLocation, useRouterLocation } from './frozen-location'

// The kept chat renders at the location it last had while pages navigate above it. This relies on
// react-router's internal location context: if an upgrade removes or reshapes it, this fails.

afterEach(cleanup)

function Where() {
  return <p data-testid="where">{useLocation().pathname}</p>
}

let go: (to: string) => void = () => undefined

function Frozen({ children }: { children: ReactNode }) {
  const live = useRouterLocation()
  const [kept] = useState<RouterLocation>(live)
  go = useNavigate()

  return (
    <>
      <p data-testid="live">{live.location.pathname}</p>
      <AtLocation location={kept}>{children}</AtLocation>
    </>
  )
}

it('gives a subtree a location of its own that later navigation does not reach', () => {
  render(
    <MemoryRouter initialEntries={['/chat']}>
      <Frozen>
        <Routes>
          <Route element={<Where />} path="/chat" />
          <Route element={<p data-testid="where">page</p>} path="*" />
        </Routes>
      </Frozen>
    </MemoryRouter>
  )

  act(() => go('/capabilities'))
  expect(screen.getByTestId('live').textContent).toBe('/capabilities')
  expect(screen.getByTestId('where').textContent).toBe('/chat')
})

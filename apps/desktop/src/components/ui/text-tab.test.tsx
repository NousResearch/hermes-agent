import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { TextTab } from './text-tab'

afterEach(cleanup)

it('announces the current button-style view without claiming tab keyboard semantics', () => {
  render(
    <>
      <TextTab active>Logs</TextTab>
      <TextTab>Settings</TextTab>
    </>
  )

  expect(screen.getByRole('button', { name: 'Logs', pressed: true })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Settings', pressed: false })).toBeTruthy()
  expect(screen.queryByRole('tab')).toBeNull()
})

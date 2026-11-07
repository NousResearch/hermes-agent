import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { SessionExportDialog } from './session-export-dialog'
afterEach(cleanup)
it('only offers saved asides and explains that unsaved side exchanges cannot be recovered', () => {
  render(
    <SessionExportDialog onClose={() => {}} target={{ sessionId: 'sample', profile: 'default', title: 'Sample' }} />
  )
  expect(screen.getByRole('checkbox', { name: 'Saved /btw asides' })).toBeTruthy()
  expect(screen.getByText(/Unsaved \/btw exchanges cannot be recovered\./)).toBeTruthy()
})

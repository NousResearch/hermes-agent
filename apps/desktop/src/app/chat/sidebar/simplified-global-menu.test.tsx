import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { SimplifiedGlobalMenu } from './simplified-global-menu'

afterEach(cleanup)

it('opens the existing Plugins manager through normal sidebar navigation', () => {
  const onNavigate = vi.fn()
  render(<SimplifiedGlobalMenu onNavigate={onNavigate} />)
  fireEvent.keyDown(screen.getByRole('button', { name: 'Hermes' }), { key: 'Enter' })
  fireEvent.click(screen.getByRole('menuitem', { name: 'Plugins' }))
  expect(onNavigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/capabilities?tab=plugins' }))
})

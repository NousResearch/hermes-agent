import { screen } from '@testing-library/dom'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('./main', () => {
  throw new Error('App chunk unavailable')
})

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('catches a failed app import at the renderer entry and shows recovery', async () => {
  const root = document.createElement('div')
  root.id = 'root'
  document.body.append(root)
  vi.spyOn(console, 'error').mockImplementation(() => {})

  const reload = vi.fn()
  vi.stubGlobal('location', { reload })

  await import('./entry')

  const button = await screen.findByRole('button', { name: 'Reload' })
  expect(console.error).toHaveBeenCalledOnce()
  expect(reload).not.toHaveBeenCalled()
  button.click()
  expect(reload).toHaveBeenCalledOnce()
})

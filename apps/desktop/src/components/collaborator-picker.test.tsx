// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { CollaboratorPicker } from './collaborator-picker'
vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))
afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

test('automatically discovers and selects a collaborator without requesting installation', async () => {
  const desktop = {
    detectCollaborators: vi.fn().mockResolvedValue([{ root: 'existing-engine', python: 'python' }]),
    selectCollaborator: vi.fn().mockResolvedValue({ ok: true }),
    continueBootstrapLocal: vi.fn()
  }
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: desktop })
  render(<CollaboratorPicker />)
  fireEvent.click(await screen.findByText('Use this Hermes'))
  await waitFor(() => expect(desktop.selectCollaborator).toHaveBeenCalledWith('existing-engine'))
  expect(desktop.continueBootstrapLocal).not.toHaveBeenCalled()
})

test('failed discovery leaves folder selection available and exposes a recoverable error', async () => {
  const desktop = {
    detectCollaborators: vi.fn().mockRejectedValueOnce(new Error('Probe failed')).mockResolvedValue([])
  }
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: desktop })
  render(<CollaboratorPicker />)
  expect(await screen.findByRole('alert')).toHaveProperty('textContent', 'Probe failed')
  const browse = screen.getByText('Choose Hermes folder')
  await waitFor(() => expect((browse.closest('button') as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(browse)
  await waitFor(() => expect(desktop.detectCollaborators).toHaveBeenCalledWith(true))
})

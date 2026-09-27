// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SkillCreate } from './skill-create'

afterEach(cleanup)

it('previews before saving to the selected connection and preserves a failed draft for retry', async () => {
  const api = vi.fn().mockRejectedValueOnce(new Error('Skill already exists')).mockResolvedValueOnce({ success: true })
  window.hermesDesktop = { ...window.hermesDesktop, api }
  const refreshed = vi.fn().mockResolvedValue(undefined)
  render(
    <I18nProvider configClient={null} initialLocale="pl">
      <SkillCreate onCreated={refreshed} profile={{ connectionId: 'local', profile: 'worker' }} scopeLabel="Worker" />
    </I18nProvider>
  )
  fireEvent.click(screen.getByText('Wklej lub utwórz skilla'))
  fireEvent.change(screen.getByLabelText('Nazwa techniczna'), { target: { value: 'raport' } })
  fireEvent.change(screen.getByPlaceholderText('Gdy proszę o tygodniowe podsumowanie pracy.'), {
    target: { value: 'Gdy proszę o raport.' }
  })
  fireEvent.change(screen.getByLabelText('Instrukcja lub pełny SKILL.md'), {
    target: { value: 'Przygotuj raport ze źródłami.' }
  })
  fireEvent.click(screen.getByText('Sprawdź podgląd'))
  expect(api).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('Dodaj skilla'))
  await screen.findByText('Skill already exists')
  expect(refreshed).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('Dodaj skilla'))
  await waitFor(() => expect(refreshed).toHaveBeenCalledOnce())
  expect(api).toHaveBeenLastCalledWith(
    expect.objectContaining({
      connectionId: 'local',
      profile: 'worker',
      method: 'POST',
      path: '/api/skills',
      body: expect.objectContaining({ name: 'raport' })
    })
  )
})

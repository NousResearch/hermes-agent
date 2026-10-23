import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $messages, $selectedStoredSessionId } from '@/store/session'
import { $temporaryDraft } from '@/store/temporary-chat'

import { TemporaryChatToggle } from './temporary-chat-toggle'

function renderToggle() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <TemporaryChatToggle disabled={false} />
    </I18nProvider>
  )
}

beforeEach(() => {
  $temporaryDraft.set(false)
  $selectedStoredSessionId.set(null)
  $messages.set([])
})

afterEach(cleanup)

test('a fresh draft offers the toggle, and it flips the temporary intent', () => {
  renderToggle()
  const button = screen.getByRole('button', { name: 'Use temporary chat' })
  expect(button.getAttribute('aria-pressed')).toBe('false')

  fireEvent.click(button)
  expect($temporaryDraft.get()).toBe(true)
  expect(screen.getByRole('button', { name: 'Exit temporary chat' }).getAttribute('aria-pressed')).toBe('true')
})

test('an existing chat cannot change mode', () => {
  act(() => $selectedStoredSessionId.set('saved'))
  renderToggle()
  expect(screen.queryByTestId('temporary-chat-toggle')).toBeNull()
})

test('a live temporary chat keeps a pressed "Temporary" pill that ends the chat after confirming', async () => {
  const { $confirmRequest } = await import('@/store/confirm')
  const { $temporaryChatEndRequests, $temporaryChats, registerTemporaryChat } = await import('@/store/temporary-chat')
  registerTemporaryChat('temp', 'rt')
  act(() => $selectedStoredSessionId.set('temp'))
  $messages.set([{ id: 'm1', parts: [], role: 'user' } as never])
  renderToggle()

  const pill = screen.getByTestId('temporary-chat-live')
  expect(pill.textContent).toContain('Temporary')
  expect(screen.queryByTestId('temporary-chat-toggle')).toBeNull()

  fireEvent.click(pill)
  await act(async () => $confirmRequest.get()?.resolve(true))
  expect($temporaryChatEndRequests.get()).toEqual(['temp'])
  $temporaryChats.set({})
  $temporaryChatEndRequests.set([])
})

import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { $workspaceIsPage } from '@/app/routes'
import { I18nProvider } from '@/i18n'
import { $butterbarItems } from '@/store/butterbar'
import { $selectedStoredSessionId } from '@/store/session'
import { $temporaryChats, $temporaryDraft, registerTemporaryChat } from '@/store/temporary-chat'

import { useTemporaryChatButterbar } from './temporary-chat-butterbar'

const open = vi.fn()

const wrapper = ({ children }: { children: ReactNode }) => (
  <I18nProvider configClient={null} initialLocale="en">
    {children}
  </I18nProvider>
)

function bar() {
  const item = $butterbarItems.get().find(entry => entry.id === 'temporary-chat')

  return item ? render(<>{item.node}</>) : null
}

beforeEach(() => {
  $temporaryChats.set({})
  $temporaryDraft.set(false)
  $selectedStoredSessionId.set(null)
  $workspaceIsPage.set(false)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

test('no temporary chat → no bar', () => {
  renderHook(() => useTemporaryChatButterbar(open), { wrapper })
  expect(bar()).toBeNull()
})

test('a temporary draft or chat on screen shows the private-mode bar', () => {
  $temporaryDraft.set(true)
  renderHook(() => useTemporaryChatButterbar(open), { wrapper })
  bar()
  expect(screen.getByTestId('temporary-chat-bar').textContent).toContain('Temporary chat')
  expect(screen.queryByText('Back to temporary chat')).toBeNull()
})

test('away from a live temporary chat (another chat or a page), the bar offers Back', () => {
  registerTemporaryChat('temp', 'rt')
  act(() => $selectedStoredSessionId.set('temp'))
  act(() => $workspaceIsPage.set(true)) // a plugin page covers the chat
  renderHook(() => useTemporaryChatButterbar(open), { wrapper })
  bar()
  act(() => screen.getByText('Back to temporary chat').click())
  expect(open).toHaveBeenCalledWith('temp')
  expect(screen.getByText('End temporary chat')).toBeTruthy()
})

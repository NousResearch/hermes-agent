import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $interfaceMode } from '@/store/interface-mode'
import { $selectedStoredSessionId } from '@/store/session'
import { $openWorkspaceSessionInNewTab } from '@/store/workspace-tab-action'

import { WorkspaceTabTitle } from './workspace-tab-title'

afterEach(() => {
  cleanup()
  $interfaceMode.set('advanced')
})

it('shows the current conversation as a preview, and returns to normal text in Advanced', () => {
  $interfaceMode.set('simple')
  const view = render(<WorkspaceTabTitle>Release plan</WorkspaceTabTitle>)
  expect(screen.getByText('Release plan').className).toContain('italic')
  view.rerender(<WorkspaceTabTitle>Meeting notes</WorkspaceTabTitle>)
  expect(screen.getByText('Meeting notes')).toBeTruthy()
  expect(screen.queryByText('Home')).toBeNull()
  act(() => $interfaceMode.set('advanced'))
  expect(screen.getByText('Meeting notes').className).not.toContain('italic')
})

it('keeps the selected preview through its action and double-click without changing its owner', () => {
  $interfaceMode.set('simple')
  $selectedStoredSessionId.set('chat-a')
  const keep = vi.fn(async () => true)
  $openWorkspaceSessionInNewTab.set(keep)
  render(<WorkspaceTabTitle>Research bot</WorkspaceTabTitle>)
  fireEvent.click(screen.getByRole('button', { name: 'Keep open' }))
  fireEvent.doubleClick(screen.getByText('Research bot'))
  expect(keep).toHaveBeenCalledTimes(2)
  expect($selectedStoredSessionId.get()).toBe('chat-a')
  $openWorkspaceSessionInNewTab.set(null)
  $selectedStoredSessionId.set(null)
})

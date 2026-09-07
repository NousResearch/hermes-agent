import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $backgroundStatusBySession } from '@/store/composer-status'
import { $sidebarRowMeta, resetSidebarView, toggleSidebarRowMeta } from '@/store/layout'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { $sidebarActivityById } from '@/store/sidebar-activity'

import { SidebarFilterMenu } from './filter-menu'
import { SidebarSessionActivity } from './session-activity'

beforeEach(() => {
  resetSidebarView()
  $sidebarRowMeta.set([])
  clearAllSessionStates()
  $backgroundStatusBySession.set({})
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $backgroundStatusBySession.set({})
  resetSidebarView()
})

const background = { id: 'process', type: 'background' as const, state: 'running' as const, title: '# Watching CI' }

function seedBackground() {
  publishSessionState('runtime', createClientSessionState('stored'))
  $backgroundStatusBySession.set({ runtime: [background] })
}

describe('sidebar activity display', () => {
  it('does not subscribe to transcript projection while disabled', () => {
    const subscribe = vi.spyOn($sidebarActivityById, 'listen')
    render(<SidebarSessionActivity sessionId="stored" />)
    expect(subscribe).not.toHaveBeenCalled()
    act(() => toggleSidebarRowMeta('activity'))
    expect(subscribe).toHaveBeenCalled()
    subscribe.mockRestore()
  })

  it('is opt-in, updates immediately, and leaves other sessions quiet', () => {
    seedBackground()

    const { container } = render(
      <>
        <SidebarSessionActivity sessionId="stored" />
        <SidebarSessionActivity sessionId="other" />
      </>
    )

    expect(container.textContent).toBe('')
    act(() => toggleSidebarRowMeta('activity'))
    expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()
    expect(container.textContent).toBe('')
    expect(container.querySelectorAll('[data-session-activity]')).toHaveLength(1)
    act(() => toggleSidebarRowMeta('activity'))
    expect(container.textContent).toBe('')
  })

  it('clears immediately on completion, before finished status rows auto-dismiss', () => {
    seedBackground()
    toggleSidebarRowMeta('activity')
    const { container } = render(<SidebarSessionActivity sessionId="stored" />)
    expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()
    act(() => $backgroundStatusBySession.set({ runtime: [{ ...background, state: 'done' }] }))
    expect(container.querySelector('[data-session-activity]')).toBeNull()
  })

  it.each(['Watching CI for the pull request', 'Waiting for the deployment health checks to finish'])(
    'reveals the full reported activity on keyboard focus: %s',
    async label => {
      seedBackground()
      $backgroundStatusBySession.set({ runtime: [{ ...background, title: `# ${label}` }] })
      toggleSidebarRowMeta('activity')
      const { container } = render(<SidebarSessionActivity sessionId="stored" />)
      const indicator = screen.getByRole('img', { name: label })
      expect(container.textContent).toBe('')
      // jsdom does not model Chromium's keyboard :focus-visible state.
      const matches = indicator.matches.bind(indicator)

      const focusVisible = vi
        .spyOn(indicator, 'matches')
        .mockImplementation(selector => selector === ':focus-visible' || matches(selector))

      fireEvent.keyDown(indicator.ownerDocument, { key: 'Tab' })
      act(() => indicator.focus())
      expect((await screen.findByRole('tooltip')).textContent).toBe(label)
      focusVisible.mockRestore()
    }
  )

  it('uses the live tool context and gives a blocking question priority', () => {
    const state = {
      ...createClientSessionState('stored'),
      busy: true,
      messages: [
        {
          id: 'reply',
          role: 'assistant' as const,
          pending: true,
          parts: [
            {
              type: 'tool-call' as const,
              toolCallId: 'watch',
              toolName: 'terminal',
              args: { context: 'Watching CI on the new PR' }
            }
          ]
        }
      ]
    }

    publishSessionState('runtime', state)
    toggleSidebarRowMeta('activity')
    render(<SidebarSessionActivity sessionId="stored" />)
    expect(screen.getByRole('img', { name: 'Watching CI on the new PR' })).toBeTruthy()
    act(() => publishSessionState('runtime', { ...state, needsInput: true }))
    expect(screen.getByRole('img', { name: 'Waiting for your answer' })).toBeTruthy()
    expect(screen.queryByRole('img', { name: 'Watching CI on the new PR' })).toBeNull()
  })

  it('falls back to the real tool title when no human context was reported', () => {
    publishSessionState('runtime', {
      ...createClientSessionState('stored'),
      busy: true,
      messages: [
        {
          id: 'reply',
          role: 'assistant',
          pending: true,
          parts: [
            {
              type: 'tool-call',
              toolCallId: 'read',
              toolName: 'read_file',
              args: { path: 'README.md' }
            }
          ]
        }
      ]
    })
    toggleSidebarRowMeta('activity')
    render(<SidebarSessionActivity sessionId="stored" />)
    expect(screen.getByRole('img', { name: 'Reading README.md' })).toBeTruthy()
  })

  it('uses an explicit command comment instead of the flattened shell preview', () => {
    publishSessionState('runtime', {
      ...createClientSessionState('stored'),
      busy: true,
      messages: [
        {
          id: 'reply',
          role: 'assistant',
          pending: true,
          parts: [
            {
              type: 'tool-call',
              toolCallId: 'watch',
              toolName: 'terminal',
              args: {
                command: '# Watching CI\ngh pr checks --watch',
                context: '# Watching CI gh pr checks --watch'
              }
            }
          ]
        }
      ]
    })
    toggleSidebarRowMeta('activity')
    render(<SidebarSessionActivity sessionId="stored" />)
    expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()
    expect(screen.queryByText(/gh pr checks/)).toBeNull()
  })

  it('can be enabled in Show without enabling inbox-style cards', async () => {
    seedBackground()
    render(
      <>
        <SidebarFilterMenu />
        <SidebarSessionActivity sessionId="stored" />
      </>
    )
    fireEvent.pointerDown(screen.getByRole('button', { name: /filter/i }), {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })
    const show = await screen.findByRole('menuitem', { name: 'Show' })
    fireEvent.keyDown(show, { key: 'ArrowRight' })
    const activity = await screen.findByRole('menuitemcheckbox', { name: 'Activity' })
    fireEvent.click(activity)
    expect($sidebarRowMeta.get()).toContain('activity')
    expect(activity.getAttribute('aria-checked')).toBe('true')
    fireEvent.keyDown(activity, { key: 'Escape' })
    expect(await screen.findByRole('img', { name: 'Watching CI' })).toBeTruthy()
  })
})

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type * as ReactRouterDom from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { assistantTextPart, textPart } from '@/lib/chat-messages'
import type * as HapticsModule from '@/lib/haptics'
import { $forkOriginNotices, $messages, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SessionForkOriginNotice } from './fork-origin-banner'

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  triggerHaptic: vi.fn()
}))

vi.mock('react-router-dom', async importOriginal => {
  const actual = await importOriginal<typeof ReactRouterDom>()

  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock('@/lib/haptics', async importOriginal => {
  const actual = await importOriginal<typeof HapticsModule>()

  return {
    ...actual,
    triggerHaptic: mocks.triggerHaptic
  }
})

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'session',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

function renderBanner(locale: 'en' | 'zh' = 'zh') {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale={locale}>
        <SessionForkOriginNotice messageId="assistant-2" storedSessionId="child-1" />
      </I18nProvider>
    </MemoryRouter>
  )
}

function resetStores() {
  $forkOriginNotices.set({})
  $messages.set([])
  $sessions.set([])
}

function chatMessage(id: string, role: ChatMessage['role'], text: string, hidden = false): ChatMessage {
  return {
    hidden,
    id,
    parts: [role === 'assistant' ? assistantTextPart(text) : textPart(text)],
    role
  }
}

beforeEach(() => {
  resetStores()
  mocks.navigate.mockReset()
  mocks.triggerHaptic.mockReset()
})

afterEach(() => {
  cleanup()
  resetStores()
  vi.restoreAllMocks()
})

describe('SessionForkOriginNotice', () => {
  it('renders only under the anchored branchable message and opens the parent live tip', () => {
    $forkOriginNotices.set({
      'child-1': { branchMessageOrdinal: 1, createdAt: 1, parentSessionId: 'parent-root-1' }
    })
    $messages.set([
      chatMessage('user-1', 'user', 'First user message'),
      chatMessage('system-1', 'system', 'not branchable'),
      chatMessage('assistant-2', 'assistant', 'Fork from here'),
      chatMessage('assistant-hidden', 'assistant', 'older hidden branch', true),
      chatMessage('assistant-3', 'assistant', 'Later response')
    ])
    $sessions.set([
      session({ id: 'child-1', title: 'Child session' }),
      session({ _lineage_root_id: 'parent-root-1', id: 'parent-tip-2', title: 'Parent session' })
    ])

    renderBanner('zh')

    expect(screen.getByText('从对话中派生')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '打开来源对话:Parent session' }))

    expect(mocks.triggerHaptic).toHaveBeenCalledWith('selection')
    expect(mocks.navigate).toHaveBeenCalledWith('/parent-tip-2')
    expect($forkOriginNotices.get()['child-1']).toBeTruthy()
  })

  it('falls back to the raw parent session id when the source session is not loaded', () => {
    $forkOriginNotices.set({
      'child-1': { branchMessageOrdinal: 1, createdAt: 1, parentSessionId: 'parent-root-1' }
    })
    $messages.set([
      chatMessage('user-1', 'user', 'First user message'),
      chatMessage('assistant-2', 'assistant', 'Fork from here'),
      chatMessage('assistant-3', 'assistant', 'Later response')
    ])
    $sessions.set([session({ id: 'child-1', title: 'Child session' })])

    renderBanner('en')

    fireEvent.click(screen.getByRole('button', { name: 'Open source conversation' }))

    expect(mocks.navigate).toHaveBeenCalledWith('/parent-root-1')
  })

  it('stays hidden for non-anchor messages', () => {
    $forkOriginNotices.set({
      'child-1': { branchMessageOrdinal: 0, createdAt: 1, parentSessionId: 'parent-root-1' }
    })
    $messages.set([
      chatMessage('user-1', 'user', 'Anchor message'),
      chatMessage('assistant-2', 'assistant', 'Not the anchor')
    ])

    render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <SessionForkOriginNotice messageId="assistant-2" storedSessionId="child-1" />
        </I18nProvider>
      </MemoryRouter>
    )

    expect(screen.queryByText('Forked from conversation')).toBeNull()
  })
})

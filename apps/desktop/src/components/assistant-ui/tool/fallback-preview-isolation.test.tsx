import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { I18nProvider } from '@/i18n'
import { $previewStatusBySession } from '@/store/preview-status'
import { $activeSessionId, $currentCwd } from '@/store/session'

import { ToolFallback } from './fallback'

vi.mock('@assistant-ui/react', async importOriginal => {
  const actual = await importOriginal<typeof import('@assistant-ui/react')>()

  return {
    ...actual,
    useAuiState: vi.fn((selector: (state: unknown) => unknown) =>
      selector({
        message: {
          id: 'assistant-b',
          status: { type: 'complete', reason: 'stop' }
        },
        thread: { isRunning: false }
      })
    )
  }
})

Element.prototype.animate = function animate() {
  return {
    cancel: () => {},
    finished: Promise.resolve()
  } as unknown as Animation
}

function tileView(runtimeId: string, cwd: string): SessionView {
  return {
    kind: 'tile',
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom(cwd),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(true),
    $model: atom(''),
    $provider: atom(''),
    $runtimeId: atom(runtimeId),
    $storedId: atom(`stored-${runtimeId}`)
  }
}

function previewToolProps(): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args: { path: '/work/b/creator.html' },
    argsText: JSON.stringify({ path: '/work/b/creator.html' }),
    isError: false,
    respondToApproval: vi.fn(),
    result: { ok: true },
    resume: vi.fn(),
    status: { type: 'complete' },
    toolCallId: 'preview-b-1',
    toolName: 'search_files',
    type: 'tool-call'
  }
}

beforeEach(() => {
  $previewStatusBySession.set({})
  $activeSessionId.set('session-a')
  $currentCwd.set('/work/a')
})

afterEach(() => {
  cleanup()
  $previewStatusBySession.set({})
  $activeSessionId.set(null)
  $currentCwd.set('')
})

describe('ToolFallback preview session isolation (#66411)', () => {
  it('records preview artifacts under the owning session view, not the global active session', async () => {
    const sessionB = tileView('session-b', '/work/b')

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={sessionB}>
          <ToolFallback {...previewToolProps()} />
        </SessionViewProvider>
      </I18nProvider>
    )

    await waitFor(() => {
      expect($previewStatusBySession.get()['session-b']?.map(item => item.id)).toEqual(['/work/b/creator.html'])
    })

    expect($previewStatusBySession.get()['session-a']).toBeUndefined()

    // Global focus flips to A while B's mounted tool row stays mounted — must
    // not re-key the artifact into A's bucket.
    $activeSessionId.set('session-a')
    $currentCwd.set('/work/a')

    await waitFor(() => {
      expect($previewStatusBySession.get()['session-b']?.map(item => item.id)).toEqual(['/work/b/creator.html'])
    })
    expect($previewStatusBySession.get()['session-a']).toBeUndefined()
  })
})

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import * as notifications from '@/store/notifications'
import {
  $activeSessionId,
  $connection,
  $selectedStoredSessionId,
  _resetSessionOwnerHintsForTests,
  setSessions
} from '@/store/session'
import { $sessionTiles, clearAllSessionStates } from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

// Normal imports throughout: real React, SessionView, stores, resolver registration,
// media helper, and i18n. Only the native save boundary is fake. No document/network I/O.
const previousDesktop = window.hermesDesktop
const save = vi.fn(async () => ({ saved: true }))
const target = '/synthetic/report.pdf'

beforeEach(() => {
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set({ mode: 'remote', connectionId: 'ambient-remote', profile: 'default' } as never)
  setApiRequestConnection('ambient-remote')
  save.mockClear()
  window.hermesDesktop = { saveGatewayFile: save } as never
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      throw new Error('Network forbidden in download UI regression')
    })
  )
  vi.spyOn(notifications, 'notifyError').mockImplementation(() => 'synthetic-notification')
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessionTiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set(null)
  setApiRequestConnection(null)
  window.hermesDesktop = previousDesktop
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function mount(storedId: string | null, runtimeId: string | null) {
  return render(
    <SessionViewProvider
      value={{
        ...PRIMARY_SESSION_VIEW,
        $cwd: atom(''),
        $storedId: atom(storedId),
        $runtimeId: atom(runtimeId)
      }}
    >
      <PreviewAttachment target={target} />
    </SessionViewProvider>
  )
}

async function download() {
  const button = await screen.findByRole('button', { name: 'Download' })
  await act(async () => {
    fireEvent.click(button)
  })
}

describe('PreviewAttachment download owner integration', () => {
  it.each(['stored', 'runtime'])(
    'click pins the local bot owner and persisted id from %s view identity',
    async identity => {
      $sessionTiles.set([
        {
          storedSessionId: 'stored',
          runtimeId: 'runtime',
          ownerRoute: { connectionId: 'local', profile: 'bot-profile' }
        }
      ] as never)
      mount(identity === 'stored' ? 'stored' : null, 'runtime')
      await download()
      expect(save).toHaveBeenCalledExactlyOnceWith({
        connectionId: 'local',
        profile: 'bot-profile',
        sessionId: 'stored',
        path: target,
        suggestedName: 'report.pdf'
      })
      expect(notifications.notifyError).not.toHaveBeenCalled()
      expect(fetch).not.toHaveBeenCalled()
    }
  )

  it('click sends the remote owner backend profile rather than its alias or foreground connection', async () => {
    $sessionTiles.set([
      {
        storedSessionId: 'stored',
        runtimeId: 'runtime',
        ownerRoute: { connectionId: 'bot-remote', profile: 'alias', targetProfile: 'bot-profile' }
      }
    ] as never)
    mount('stored', 'runtime')
    await download()
    expect(save).toHaveBeenCalledExactlyOnceWith({
      connectionId: 'bot-remote',
      profile: 'bot-profile',
      sessionId: 'stored',
      path: target,
      suggestedName: 'report.pdf'
    })
    expect(notifications.notifyError).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()
  })

  it('unknown ownership reports failure without saving or making a network request', async () => {
    mount('unknown', null)
    await download()
    expect(save).not.toHaveBeenCalled()
    expect(notifications.notifyError).toHaveBeenCalledOnce()
    expect(fetch).not.toHaveBeenCalled()
  })
})

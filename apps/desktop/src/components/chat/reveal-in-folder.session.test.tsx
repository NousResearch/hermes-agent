import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { RevealInFolderTrigger } from './reveal-in-folder'

describe('RevealInFolderTrigger session routing', () => {
  afterEach(() => {
    cleanup()
    _resetSessionOwnerHintsForTests({ storage: true })
    $connection.set(null)
  })

  it('shows local actions for the viewed local session while the ambient connection is remote', () => {
    setSessionOwnerHint('stored-local', { connectionId: 'local', mode: 'local', profile: 'default' })
    $connection.set({ mode: 'remote' } as never)

    render(
      <SessionViewProvider value={{ ...PRIMARY_SESSION_VIEW, $storedId: atom('stored-local') }}>
        <RevealInFolderTrigger path="C:\\work\\report.md">
          <span>report.md</span>
        </RevealInFolderTrigger>
      </SessionViewProvider>
    )

    expect(screen.getByRole('button', { name: 'File actions' })).toBeTruthy()
  })

  it('hides local actions for the viewed remote session while the ambient connection is local', () => {
    setSessionOwnerHint('stored-remote', { connectionId: 'ssh', mode: 'remote', profile: 'default' })
    $connection.set({ mode: 'local' } as never)

    render(
      <SessionViewProvider value={{ ...PRIMARY_SESSION_VIEW, $storedId: atom('stored-remote') }}>
        <RevealInFolderTrigger path="/srv/report.md">
          <span>report.md</span>
        </RevealInFolderTrigger>
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'File actions' })).toBeNull()
    expect(screen.getByText('report.md')).toBeTruthy()
  })
})

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $connection, $sessions } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, dropSessionState, recordSessionEventScope } from '@/store/session-states'

import { PreviewAttachment } from './preview-attachment'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(resolvePromise => {
    resolve = resolvePromise
  })

  return { promise, resolve }
}

function sessionView(
  cwd: string,
  {
    kind = 'primary',
    runtimeId = null,
    storedId = null
  }: Partial<{ kind: SessionView['kind']; runtimeId: null | string; storedId: null | string }> = {}
): SessionView {
  return {
    ...({} as SessionView),
    $cwd: atom(cwd),
    $runtimeId: atom(runtimeId),
    $storedId: atom(storedId),
    kind
  }
}

describe('PreviewAttachment local file actions', () => {
  const normalizePreviewTarget = vi.fn(async () => null)
  const openExternal = vi.fn(async () => undefined)
  const revealPath = vi.fn(async () => true)
  let originalDesktop: typeof window.hermesDesktop

  beforeEach(() => {
    originalDesktop = window.hermesDesktop
    normalizePreviewTarget.mockReset()
    normalizePreviewTarget.mockResolvedValue(null)
    openExternal.mockClear()
    revealPath.mockClear()
    clearNotifications()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { normalizePreviewTarget, openExternal, revealPath }
    })
    $connection.set({ mode: 'local' } as never)
    $connectionsRegistry.set(null)
    $sessions.set([])
    $sessionTiles.set([])
    clearAllSessionStates()
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    $connection.set(null)
    $connectionsRegistry.set(null)
    $sessions.set([])
    $sessionTiles.set([])
    clearAllSessionStates()
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: originalDesktop
    })
  })

  it('opens and reveals a relative file for the primary explicit-local session', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'status #1.txt',
      path: '/workspace/reports/status #1.txt',
      previewKind: 'text',
      source: 'reports/status #1.txt',
      url: 'file:///workspace/reports/status%20%231.txt'
    } as never)

    render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target="reports/status #1.txt" />
      </SessionViewProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Open file' }))

    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('file:///workspace/reports/status%20%231.txt'))

    fireEvent.click(
      screen.getByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    )

    await waitFor(() => expect(revealPath).toHaveBeenCalledWith('/workspace/reports/status #1.txt'))
  })

  it('lets the four-action group wrap instead of overflowing narrow cards', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/workspace/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///workspace/report.txt'
    } as never)

    render(<PreviewAttachment target="/workspace/report.txt" />)

    const actions = (await screen.findByRole('button', { name: 'Open file' })).parentElement
    const classes = actions?.className.split(/\s+/) ?? []

    expect(classes).toContain('min-w-0')
    expect(classes).toContain('flex-wrap')
    expect(classes).toContain('justify-end')
    expect(classes).not.toContain('shrink-0')
  })

  it('allows exact local ownership under an ambient remote connection', async () => {
    $connection.set({ mode: 'remote', profile: 'remote-work' } as never)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'local', mode: 'local', profile: 'default' },
        runtimeId: 'rt-local-under-remote',
        storedSessionId: 'stored-local-under-remote'
      }
    ] as never)
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      path: '/workspace/report.txt',
      url: 'file:///workspace/report.txt'
    } as never)

    render(
      <SessionViewProvider
        value={sessionView('/workspace', {
          kind: 'tile',
          runtimeId: 'rt-local-under-remote',
          storedId: 'stored-local-under-remote'
        })}
      >
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Open file' }))
    expect(openExternal).toHaveBeenCalledWith('file:///workspace/report.txt')
    fireEvent.click(
      screen.getByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    )
    expect(revealPath).toHaveBeenCalledWith('/workspace/report.txt')
  })

  it('hides local actions for a remote-owned tile under an ambient local connection', () => {
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'homelab', mode: 'remote', profile: 'default' },
        runtimeId: 'rt-remote',
        storedSessionId: 'stored-remote'
      }
    ] as never)

    render(
      <SessionViewProvider
        value={sessionView('/remote/work', { kind: 'tile', runtimeId: 'rt-remote', storedId: 'stored-remote' })}
      >
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })

  it('hides local actions for a remote-owned primary session under an ambient local connection', () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/remote/work/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///remote/work/report.txt'
    } as never)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'homelab', mode: 'remote', profile: 'default' },
        runtimeId: 'rt-remote-primary',
        storedSessionId: 'stored-remote-primary'
      }
    ] as never)

    render(
      <SessionViewProvider
        value={sessionView('/remote/work', {
          runtimeId: 'rt-remote-primary',
          storedId: 'stored-remote-primary'
        })}
      >
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
  })

  it('fails closed when a connection-tagged session row belongs to a registered remote gateway', () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/remote/work/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///remote/work/report.txt'
    } as never)
    $connectionsRegistry.set({
      activeId: 'local',
      connections: [
        { id: 'local', kind: 'local', name: 'Local' },
        { id: 'homelab', kind: 'remote', name: 'Home lab' }
      ],
      version: 1
    } as never)
    $sessions.set([
      {
        connection_id: 'homelab',
        id: 'stored-remote-row',
        profile: 'default'
      }
    ] as never)

    render(
      <SessionViewProvider value={sessionView('/remote/work', { storedId: 'stored-remote-row' })}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
  })

  it('fails closed for a profile-only owner in registry topology', () => {
    $connectionsRegistry.set({
      activeId: 'local',
      connections: [{ id: 'local', kind: 'local', name: 'Local' }],
      version: 1
    } as never)
    $sessions.set([{ id: 'stored-profile-only', profile: 'default' }] as never)

    render(
      <SessionViewProvider value={sessionView('/workspace', { storedId: 'stored-profile-only' })}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
  })

  it('fails closed for a mode-less tile owner registered to a remote gateway', () => {
    $connectionsRegistry.set({
      activeId: 'local',
      connections: [
        { id: 'local', kind: 'local', name: 'Local' },
        { id: 'homelab', kind: 'remote', name: 'Home lab' }
      ],
      version: 1
    } as never)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'homelab', profile: 'default' },
        runtimeId: 'rt-mode-less-remote',
        storedSessionId: 'stored-mode-less-remote'
      }
    ] as never)

    render(
      <SessionViewProvider
        value={sessionView('/remote/work', {
          kind: 'tile',
          runtimeId: 'rt-mode-less-remote',
          storedId: 'stored-mode-less-remote'
        })}
      >
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(normalizePreviewTarget).not.toHaveBeenCalled()
  })

  it('removes local actions when the owning registry connection changes from local to remote', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/workspace/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///workspace/report.txt'
    } as never)
    $sessions.set([
      {
        connection_id: 'edge',
        id: 'stored-registry-change',
        profile: 'default'
      }
    ] as never)
    $connectionsRegistry.set({
      activeId: 'local',
      connections: [{ id: 'edge', kind: 'local', name: 'Edge' }],
      version: 1
    } as never)

    render(
      <SessionViewProvider value={sessionView('/workspace', { storedId: 'stored-registry-change' })}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()

    act(() => {
      $connectionsRegistry.set({
        activeId: 'local',
        connections: [{ id: 'edge', kind: 'remote', name: 'Edge' }],
        version: 1
      } as never)
    })

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull())
  })

  it('removes local actions when a mounted session row changes to a remote owner', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/workspace/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///workspace/report.txt'
    } as never)
    $connectionsRegistry.set({
      activeId: 'local',
      connections: [
        { id: 'local-edge', kind: 'local', name: 'Local edge' },
        { id: 'remote-edge', kind: 'remote', name: 'Remote edge' }
      ],
      version: 1
    } as never)
    $sessions.set([
      {
        connection_id: 'local-edge',
        id: 'stored-row-change',
        profile: 'default'
      }
    ] as never)

    render(
      <SessionViewProvider value={sessionView('/workspace', { storedId: 'stored-row-change' })}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()

    act(() => {
      $sessions.set([
        {
          connection_id: 'remote-edge',
          id: 'stored-row-change',
          profile: 'default'
        }
      ] as never)
    })

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull())
  })

  it('removes local actions when a mounted tile owner changes from local to remote', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/workspace/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///workspace/report.txt'
    } as never)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'local', mode: 'local', profile: 'default' },
        runtimeId: 'rt-owner-change',
        storedSessionId: 'stored-owner-change'
      }
    ] as never)

    render(
      <SessionViewProvider
        value={sessionView('/workspace', {
          kind: 'tile',
          runtimeId: 'rt-owner-change',
          storedId: 'stored-owner-change'
        })}
      >
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()

    act(() => {
      $sessionTiles.set([
        {
          ownerRoute: { connectionId: 'homelab', mode: 'remote', profile: 'default' },
          runtimeId: 'rt-owner-change',
          storedSessionId: 'stored-owner-change'
        }
      ] as never)
    })

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull())
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
  })

  it('reacts when the exact runtime event owner changes or is removed', async () => {
    $connectionsRegistry.set({
      activeId: 'local-edge',
      connections: [
        { id: 'local-edge', kind: 'local', name: 'Local edge' },
        { id: 'remote-edge', kind: 'remote', name: 'Remote edge' }
      ],
      version: 1
    } as never)
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      path: '/workspace/report.txt',
      url: 'file:///workspace/report.txt'
    } as never)
    recordSessionEventScope({ connectionId: 'local-edge', profile: 'default', session_id: 'rt-event-owner' })

    render(
      <SessionViewProvider value={sessionView('/workspace', { runtimeId: 'rt-event-owner' })}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()

    act(() =>
      recordSessionEventScope({ connectionId: 'remote-edge', profile: 'default', session_id: 'rt-event-owner' })
    )
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull())

    act(() => recordSessionEventScope({ connectionId: 'local-edge', profile: 'default', session_id: 'rt-event-owner' }))
    expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()

    act(() => dropSessionState('rt-event-owner'))
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull())
  })

  it('fails closed for a tile whose owner is unknown', () => {
    render(
      <SessionViewProvider value={sessionView('/workspace', { kind: 'tile', runtimeId: 'rt-unknown' })}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })

  it.each([null, {}])('fails closed when the primary connection is null or unresolved', connection => {
    $connection.set(connection as never)

    render(<PreviewAttachment target="/workspace/report.txt" />)

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })

  it.each([
    {
      normalized: {
        kind: 'url',
        label: 'example.com',
        source: 'https://example.com/report.txt',
        url: 'https://example.com/report.txt'
      },
      target: 'https://example.com/report.txt'
    },
    {
      normalized: {
        kind: 'file',
        label: 'relative.txt',
        path: 'relative.txt',
        previewKind: 'text',
        source: 'relative.txt',
        url: 'file:///relative.txt'
      },
      target: 'invalid-relative-result'
    }
  ])('hides local actions when $target does not resolve to an absolute local file', async ({ normalized, target }) => {
    normalizePreviewTarget.mockResolvedValue(normalized as never)

    render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target={target} />
      </SessionViewProvider>
    )

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledWith(target, '/workspace'))
    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })

  it.each(['null', 'rejection'] as const)('hides local actions when the native resolver returns %s', async result => {
    if (result === 'null') {
      normalizePreviewTarget.mockResolvedValue(null)
    } else {
      normalizePreviewTarget.mockRejectedValue(new Error('Native file lookup failed'))
    }

    render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target="missing-report.txt" />
      </SessionViewProvider>
    )

    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledWith('missing-report.txt', '/workspace'))
    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })

  it('ignores a deferred resolver from an earlier local mode after switching away and back', async () => {
    const stale = deferred<never>()
    const current = deferred<never>()
    normalizePreviewTarget.mockReturnValueOnce(stale.promise).mockReturnValueOnce(current.promise)

    render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(1))
    act(() => $connection.set({ mode: 'remote', profile: 'remote-work' } as never))
    act(() => $connection.set({ mode: 'local' } as never))
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(2))

    await act(async () => {
      stale.resolve({
        kind: 'file',
        label: 'stale.txt',
        path: '/workspace/stale.txt',
        previewKind: 'text',
        source: 'report.txt',
        url: 'file:///workspace/stale.txt'
      } as never)
    })

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(openExternal).not.toHaveBeenCalled()
    expect(revealPath).not.toHaveBeenCalled()

    await act(async () => {
      current.resolve({
        kind: 'file',
        label: 'report.txt',
        path: '/workspace/report.txt',
        previewKind: 'text',
        source: 'report.txt',
        url: 'file:///workspace/report.txt'
      } as never)
    })

    await waitFor(() => expect(screen.getByRole('button', { name: 'Open file' })).toBeTruthy())
  })

  it('ignores a deferred resolver after the target and owning cwd rerender', async () => {
    const stale = deferred<never>()
    const current = deferred<never>()
    normalizePreviewTarget.mockReturnValueOnce(stale.promise).mockReturnValueOnce(current.promise)

    const view = render(
      <SessionViewProvider value={sessionView('/old-cwd')}>
        <PreviewAttachment target="old.txt" />
      </SessionViewProvider>
    )

    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(1))

    view.rerender(
      <SessionViewProvider value={sessionView('/new-cwd')}>
        <PreviewAttachment target="new.txt" />
      </SessionViewProvider>
    )
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(2))

    await act(async () => {
      stale.resolve({
        kind: 'file',
        label: 'old.txt',
        path: '/old-cwd/old.txt',
        previewKind: 'text',
        source: 'old.txt',
        url: 'file:///old-cwd/old.txt'
      } as never)
    })

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()

    await act(async () => {
      current.resolve({
        kind: 'file',
        label: 'new.txt',
        path: '/new-cwd/new.txt',
        previewKind: 'text',
        source: 'new.txt',
        url: 'file:///new-cwd/new.txt'
      } as never)
    })

    fireEvent.click(await screen.findByRole('button', { name: 'Open file' }))
    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('file:///new-cwd/new.txt'))
    expect(openExternal).not.toHaveBeenCalledWith('file:///old-cwd/old.txt')
  })

  it('ignores a deferred resolver after unmount', async () => {
    const pending = deferred<never>()
    normalizePreviewTarget.mockReturnValueOnce(pending.promise)

    const view = render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(1))
    view.unmount()

    await act(async () => {
      pending.resolve({
        kind: 'file',
        label: 'report.txt',
        path: '/workspace/report.txt',
        previewKind: 'text',
        source: 'report.txt',
        url: 'file:///workspace/report.txt'
      } as never)
    })

    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(openExternal).not.toHaveBeenCalled()
    expect(revealPath).not.toHaveBeenCalled()
  })

  it('surfaces a reveal failure instead of silently succeeding', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/workspace/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///workspace/report.txt'
    } as never)
    revealPath.mockResolvedValueOnce(false)

    render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    fireEvent.click(
      await screen.findByRole('button', {
        name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/
      })
    )

    await waitFor(() => {
      expect($notifications.get()[0]).toMatchObject({
        kind: 'error',
        message: 'Could not reveal local file: /workspace/report.txt'
      })
    })
  })

  it('rechecks the local connection before dispatching a detached native action', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/workspace/report.txt',
      previewKind: 'text',
      source: 'report.txt',
      url: 'file:///workspace/report.txt'
    } as never)

    render(
      <SessionViewProvider value={sessionView('/workspace')}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )

    const openButton = await screen.findByRole('button', { name: 'Open file' })

    act(() => {
      $connection.set({ mode: 'remote', profile: 'remote-work' } as never)
      openButton.click()
    })

    expect(openExternal).not.toHaveBeenCalled()
  })

  it.each(['runtime', 'stored', 'cwd'] as const)(
    'rejects detached native actions when the live %s changes before React rerenders',
    async changed => {
      const $runtimeId = atom<null | string>('rt-local')
      const $storedId = atom<null | string>('stored-local')
      const $cwd = atom('/workspace')
      const view = { ...sessionView('/workspace'), $cwd, $runtimeId, $storedId }
      $sessionTiles.set([
        {
          ownerRoute: { connectionId: 'local', mode: 'local', profile: 'default' },
          runtimeId: 'rt-local',
          storedSessionId: 'stored-local'
        },
        {
          ownerRoute: { connectionId: 'homelab', mode: 'remote', profile: 'default' },
          runtimeId: 'rt-remote',
          storedSessionId: 'stored-remote'
        }
      ] as never)
      normalizePreviewTarget.mockResolvedValue({
        kind: 'file',
        path: '/workspace/report.txt',
        url: 'file:///workspace/report.txt'
      } as never)

      render(
        <SessionViewProvider value={view}>
          <PreviewAttachment target="report.txt" />
        </SessionViewProvider>
      )
      const openButton = await screen.findByRole('button', { name: 'Open file' })

      const revealButton = screen.getByRole('button', {
        name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/
      })

      await act(async () => {
        if (changed === 'runtime') {
          $runtimeId.set('rt-remote')
        }

        if (changed === 'stored') {
          $storedId.set('stored-remote')
        }

        if (changed === 'cwd') {
          $cwd.set('/another-workspace')
        }

        openButton.click()
        revealButton.click()
      })

      expect(openExternal).not.toHaveBeenCalled()
      expect(revealPath).not.toHaveBeenCalled()
    }
  )

  it('discards a pending local resolution when session identity changes with the same cwd and target', async () => {
    const stale = deferred<never>()
    const current = deferred<never>()
    const $storedId = atom<null | string>('stored-local-a')
    const view = { ...sessionView('/workspace'), $storedId }
    $sessions.set([
      { id: 'stored-local-a', profile: 'default' },
      { id: 'stored-local-b', profile: 'default' }
    ] as never)
    normalizePreviewTarget.mockReturnValueOnce(stale.promise).mockReturnValueOnce(current.promise)

    render(
      <SessionViewProvider value={view}>
        <PreviewAttachment target="report.txt" />
      </SessionViewProvider>
    )
    await waitFor(() => expect(normalizePreviewTarget).toHaveBeenCalledTimes(1))
    act(() => $storedId.set('stored-local-b'))
    await act(async () => {
      stale.resolve({ kind: 'file', path: '/workspace/stale.txt', url: 'file:///workspace/stale.txt' } as never)
    })
    expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull()
    expect(normalizePreviewTarget).toHaveBeenCalledTimes(2)

    await act(async () => {
      current.resolve({ kind: 'file', path: '/workspace/current.txt', url: 'file:///workspace/current.txt' } as never)
    })
    fireEvent.click(await screen.findByRole('button', { name: 'Open file' }))
    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('file:///workspace/current.txt'))
  })

  it('hides desktop file actions after switching to a remote path without removing download or preview', async () => {
    normalizePreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'report.txt',
      path: '/remote/work/report.txt',
      previewKind: 'text',
      source: '/remote/work/report.txt',
      url: 'file:///remote/work/report.txt'
    } as never)

    render(<PreviewAttachment target="/remote/work/report.txt" />)
    expect(await screen.findByRole('button', { name: 'Open file' })).toBeTruthy()

    act(() => $connection.set({ mode: 'remote', profile: 'remote-work' } as never))

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open file' })).toBeNull())
    expect(
      screen.queryByRole('button', { name: /Reveal in Finder|Reveal in File Explorer|Open containing folder/ })
    ).toBeNull()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })
})

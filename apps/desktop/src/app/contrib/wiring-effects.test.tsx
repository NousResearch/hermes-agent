import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type OnSessionsRefreshed, useGatewayScopeRefresh } from './use-gateway-scope-refresh'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

function mountScopeRefresh(props: {
  connectionId: null | string
  profile: string
  refreshCurrentModel: (force: boolean) => Promise<unknown>
  refreshHermesConfig: (force: boolean) => Promise<unknown>
  refreshSessions: () => Promise<unknown>
  restoreMostRecentSession: () => void
}) {
  const container = document.createElement('div')
  document.body.append(container)
  const root = createRoot(container)

  const listeners = new Set<Parameters<OnSessionsRefreshed>[0]>()

  const onSessionsRefreshed: OnSessionsRefreshed = listener => {
    listeners.add(listener)

    return () => {
      listeners.delete(listener)
    }
  }

  function Harness(currentProps: typeof props) {
    useGatewayScopeRefresh(
      currentProps.connectionId,
      currentProps.profile,
      currentProps.refreshCurrentModel,
      currentProps.refreshHermesConfig,
      async () => {
        await currentProps.refreshSessions()

        for (const listener of listeners) {
          listener({ connectionId: currentProps.connectionId, profile: currentProps.profile })
        }
      },
      currentProps.restoreMostRecentSession,
      onSessionsRefreshed,
      () => 'unchanged-intent'
    )

    return null
  }

  act(() => root.render(<Harness {...props} />))

  return { container, root, rerender: (nextProps: typeof props) => act(() => root.render(<Harness {...nextProps} />)) }
}

describe('useGatewayScopeRefresh session recovery', () => {
  let mounted: { container: HTMLDivElement; root: Root } | undefined

  afterEach(() => {
    act(() => mounted?.root.unmount())
    mounted?.container.remove()
    mounted = undefined
  })

  it('does not restore a stale B result after switching B back to A', async () => {
    const refreshes = [deferred<void>(), deferred<void>()]

    const refreshSessions = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(() => refreshes[0].promise)
      .mockImplementationOnce(() => refreshes[1].promise)

    const restoreMostRecentSession = vi.fn()
    const refreshCurrentModel = vi.fn(async () => undefined)
    const refreshHermesConfig = vi.fn(async () => undefined)

    const initialProps = {
      connectionId: 'A',
      profile: 'default',
      refreshCurrentModel,
      refreshHermesConfig,
      refreshSessions,
      restoreMostRecentSession
    }

    const view = mountScopeRefresh(initialProps)
    mounted = view

    view.rerender({ ...initialProps, connectionId: 'B' })
    view.rerender({ ...initialProps, connectionId: 'A' })

    await act(async () => {
      refreshes[0].resolve()
      await Promise.resolve()
    })
    expect(restoreMostRecentSession).not.toHaveBeenCalled()

    await act(async () => {
      refreshes[1].resolve()
      await Promise.resolve()
    })
    expect(restoreMostRecentSession).toHaveBeenCalledOnce()
  })

  it('swallows a failed session refresh without restoring', async () => {
    const refresh = deferred<void>()
    const restoreMostRecentSession = vi.fn()
    const refreshCurrentModel = vi.fn(async () => undefined)
    const refreshHermesConfig = vi.fn(async () => undefined)

    const initialProps = {
      connectionId: null,
      profile: 'default',
      refreshCurrentModel,
      refreshHermesConfig,
      refreshSessions: () => refresh.promise,
      restoreMostRecentSession
    }

    const view = mountScopeRefresh(initialProps)
    mounted = view

    view.rerender({ ...initialProps, profile: 'worker' })

    await act(async () => {
      refresh.reject(new Error('session list unavailable'))
      await expect(refresh.promise).rejects.toThrow('session list unavailable')
    })
    expect(restoreMostRecentSession).not.toHaveBeenCalled()
  })

  it('uses the latest callback identity while a scope refresh is pending', async () => {
    const refresh = deferred<void>()
    const firstRestore = vi.fn()
    const latestRestore = vi.fn()
    const refreshCurrentModel = vi.fn(async () => undefined)
    const refreshHermesConfig = vi.fn(async () => undefined)

    const initialProps = {
      connectionId: 'A',
      profile: 'default',
      refreshCurrentModel,
      refreshHermesConfig,
      refreshSessions: () => refresh.promise,
      restoreMostRecentSession: firstRestore
    }

    const view = mountScopeRefresh(initialProps)
    mounted = view

    view.rerender({ ...initialProps, connectionId: 'B' })
    view.rerender({ ...initialProps, connectionId: 'B', restoreMostRecentSession: latestRestore })

    await act(async () => {
      refresh.resolve()
      await refresh.promise
    })

    expect(firstRestore).not.toHaveBeenCalled()
    expect(latestRestore).toHaveBeenCalledOnce()
  })
})

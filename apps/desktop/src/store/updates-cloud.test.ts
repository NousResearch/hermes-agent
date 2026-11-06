// @vitest-environment jsdom
// The update store's own entry points refuse a Cloud-owned target, so no button,
// toast or flow left anywhere can start an update Hermes Cloud owns.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

const checkHermesUpdateSpy = vi.fn()
const updateHermesSpy = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  checkHermesUpdate: (...args: unknown[]) => checkHermesUpdateSpy(...args),
  getActionStatus: vi.fn(),
  updateHermes: (...args: unknown[]) => updateHermesSpy(...args)
}))
const notifySpy = vi.fn()

vi.mock('@/store/notifications', () => ({ dismissNotification: vi.fn(), notify: (...a: unknown[]) => notifySpy(...a) }))

const updates = await import('./updates')
const { setConnection } = await import('./session')

const clientCheck = vi.fn().mockResolvedValue({ supported: true, behind: 2, targetSha: 'a' })
const clientApply = vi.fn().mockResolvedValue({ ok: true })

const remote = (baseUrl: string): HermesConnection => ({
  baseUrl,
  isFullscreen: false,
  logs: [],
  mode: 'remote',
  nativeOverlayWidth: 0,
  remoteKind: 'cloud',
  token: 't',
  windowButtonPosition: null,
  wsUrl: baseUrl.replace('https', 'wss')
})

beforeEach(() => {
  vi.clearAllMocks()
  checkHermesUpdateSpy.mockResolvedValue({ behind: 1, can_apply: true, current_version: '1', update_available: true })
  updateHermesSpy.mockResolvedValue({ ok: true, name: 'hermes-update' })
  window.hermesDesktop = { updates: { apply: clientApply, check: clientCheck, onProgress: () => () => {} } } as never
  updates.$updateOverlayOpen.set(false)
})

afterEach(() => {
  vi.unstubAllGlobals()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
})

describe('update entry points under Hermes Cloud', () => {
  it('a Webapp tab served by a Cloud agent checks, applies and opens nothing', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.stubGlobal('location', new URL('https://lab--app.agents.nousresearch.com/'))
    setConnection(remote('https://lab--app.agents.nousresearch.com'))

    await updates.checkUpdates({ force: true })
    await updates.checkBackendUpdates({ force: true })
    expect(await updates.applyUpdates()).toMatchObject({ ok: false, error: 'cloud-managed' })
    expect(await updates.applyBackendUpdate()).toMatchObject({ ok: false, error: 'cloud-managed' })
    updates.openUpdateOverlayFor('client')
    updates.startActiveUpdate()
    updates.requestActiveUpdate()

    expect(clientCheck).not.toHaveBeenCalled()
    expect(clientApply).not.toHaveBeenCalled()
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()
    expect(updateHermesSpy).not.toHaveBeenCalled()
    expect(updates.$updateOverlayOpen.get()).toBe(false)
  })

  it('native Desktop connected to a Cloud agent still updates its own app, never the agent', async () => {
    setConnection(remote('https://lab.agents.nousresearch.com'))

    await updates.checkBackendUpdates({ force: true })
    expect(await updates.applyBackendUpdate()).toMatchObject({ ok: false, error: 'cloud-managed' })
    expect(checkHermesUpdateSpy).not.toHaveBeenCalled()
    expect(updateHermesSpy).not.toHaveBeenCalled()

    await updates.checkUpdates({ force: true })
    expect(clientCheck).toHaveBeenCalled()
    updates.openUpdateOverlayFor('client')
    expect(updates.$updateOverlayOpen.get()).toBe(true)
  })

  it('the generic Update command on Desktop connected to Cloud updates this app and skips the agent', async () => {
    setConnection(remote('https://lab.agents.nousresearch.com'))
    updates.$updateStatus.set({ supported: true, behind: 2, targetSha: 'a' })

    updates.startActiveUpdate()

    await vi.waitFor(() => expect(clientApply).toHaveBeenCalled())
    expect(updates.$updateOverlayTarget.get()).toBe('client')
    expect(updateHermesSpy).not.toHaveBeenCalled()
  })

  it('a Cloud-served tab running older code than its agent offers Reload, not an updater', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    const reload = vi.fn()
    vi.stubGlobal('location', Object.assign(new URL('https://lab--app.agents.nousresearch.com/'), { reload }))

    updates.reportBackendContract(updates.REQUIRED_BACKEND_CONTRACT + 1)
    const toast = notifySpy.mock.calls.at(-1)?.[0] as { action: { onClick: () => void } }
    toast.action.onClick()

    expect(reload).toHaveBeenCalled()
    expect(updates.$updateOverlayOpen.get()).toBe(false)
  })
})

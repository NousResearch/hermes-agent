import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { requestGateway, installAgentPlugin, loadAgentPlugins, discoverRuntimePlugins, notify } = vi.hoisted(() => ({
  requestGateway: vi.fn(),
  installAgentPlugin: vi.fn(),
  loadAgentPlugins: vi.fn(async () => undefined),
  discoverRuntimePlugins: vi.fn<() => Promise<void>>(async () => undefined),
  notify: vi.fn()
}))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway })
}))

vi.mock('@/store/agent-plugins', () => ({ installAgentPlugin, loadAgentPlugins }))
vi.mock('@/contrib/runtime-loader', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  discoverRuntimePlugins
}))
vi.mock('@/store/notifications', () => ({ notify, notifyError: vi.fn() }))
// Same rounds, short waits: the real 3 × 2 s outlast an in-flight scan; here
// they would only slow the "rescan dropped" and "never published" cases.
vi.mock('./plugin-install-plan', async importOriginal => {
  const plan = await importOriginal<typeof Plan>()

  return {
    ...plan,
    settleUnifiedDesktopPluginId: (
      rescan: () => Promise<void>,
      records: Parameters<typeof plan.settleUnifiedDesktopPluginId>[1],
      entryFile: string
    ) => plan.settleUnifiedDesktopPluginId(rescan, records, entryFile, plan.UNIFIED_RECORD_ATTEMPTS, 40)
  }
})

import { $pluginDecisions, $pluginRecords, publishPlugin } from '@/contrib/plugins-store'
import { I18nProvider } from '@/i18n'
import { closePluginInstallRequest, openPluginInstallRequest } from '@/store/plugin-install-request'
import { $connection } from '@/store/session'

import { PluginInstallModal } from './plugin-install-modal'
import type * as Plan from './plugin-install-plan'

const REPO = 'https://github.com/example/word-count'
const PLUGINS_ROOT = '/home/u/.hermes/plugins'
const DESKTOP_ROOT = '/home/u/.hermes/desktop-plugins'
const UNIFIED_ENTRY = `${PLUGINS_ROOT}/word-count/desktop/plugin.js`

const hybridProbe = {
  ok: true,
  agent: true,
  desktop: true,
  agentName: 'word-count',
  desktopName: 'word-count',
  desktopSourceSubdir: 'desktop' as '.' | 'desktop' | null,
  warnings: [] as string[],
  insecure: false
}

const probePluginRepo = vi.fn(async () => hybridProbe)
const installDesktopPlugin = vi.fn(async () => ({ ok: true, pluginName: 'word-count' }))

/** What the agent install leaves under plugins/ — the unified desktop half is
 *  present only after a successful (or "already installed") agent install. */
let unifiedHalfOnDisk = false

/** A desktop-plugins/word-count/plugin.js left by an install made before this fix. */
let standaloneOnDisk = false

const readDir = vi.fn(async (dirPath: string) => {
  const entry = (name: string, isDirectory: boolean) => ({ name, path: `${dirPath}/${name}`, isDirectory })

  if (dirPath === DESKTOP_ROOT) {
    return { entries: standaloneOnDisk ? [entry('word-count', true)] : [] }
  }

  if (dirPath === `${DESKTOP_ROOT}/word-count`) {
    return { entries: [entry('plugin.js', false)] }
  }

  if (dirPath === PLUGINS_ROOT) {
    return { entries: unifiedHalfOnDisk ? [entry('word-count', true)] : [] }
  }

  if (dirPath === `${PLUGINS_ROOT}/word-count`) {
    return { entries: [entry('plugin.yaml', false), entry('desktop', true)] }
  }

  if (dirPath === `${PLUGINS_ROOT}/word-count/desktop`) {
    return { entries: [entry('plugin.js', false)] }
  }

  return { entries: [], error: 'ENOENT' }
})

/** What the disk door publishes once it has scanned the unified half. */
function publishUnifiedHalf() {
  publishPlugin({
    id: 'word-count',
    name: 'Word count',
    kind: 'disk',
    status: 'disabled',
    file: UNIFIED_ENTRY
  })
}

function setConnection(mode: 'local' | 'remote' | undefined) {
  $connection.set({
    baseUrl: 'http://127.0.0.1:8642',
    isFullscreen: false,
    logs: [],
    mode,
    nativeOverlayWidth: 0,
    token: 't',
    windowButtonPosition: null,
    wsUrl: 'ws://127.0.0.1:8642'
  })
}

function renderModal() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <I18nProvider configClient={null} initialLocale="en">
        <PluginInstallModal />
      </I18nProvider>
    </MemoryRouter>
  )
}

async function openModal() {
  renderModal()

  await act(async () => {
    openPluginInstallRequest({ repo: REPO })
  })

  const install = await screen.findByRole('button', { name: 'Install' })

  await waitFor(() => expect(install.hasAttribute('disabled')).toBe(false))

  return install
}

async function clickInstall(install: HTMLElement) {
  fireEvent.click(install)

  await waitFor(() => expect(loadAgentPlugins).toHaveBeenCalled())
}

const successToast = (fragment: string) =>
  expect.objectContaining({ kind: 'success', message: expect.stringContaining(fragment) })

beforeEach(() => {
  requestGateway.mockReset()
  installAgentPlugin.mockReset()
  installAgentPlugin.mockImplementation(async () => {
    unifiedHalfOnDisk = true

    return { ok: true, pluginName: 'word-count' }
  })
  loadAgentPlugins.mockClear()
  discoverRuntimePlugins.mockReset()
  discoverRuntimePlugins.mockImplementation(async () => {
    publishUnifiedHalf()
  })
  notify.mockClear()
  probePluginRepo.mockReset()
  probePluginRepo.mockImplementation(async () => hybridProbe)
  installDesktopPlugin.mockClear()
  readDir.mockClear()
  unifiedHalfOnDisk = false
  standaloneOnDisk = false
  $pluginRecords.set({})
  $pluginDecisions.set({})
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    probePluginRepo,
    installDesktopPlugin,
    readDir,
    agentPluginsRoot: async () => PLUGINS_ROOT,
    desktopPluginsRoot: async () => DESKTOP_ROOT
  }
})

afterEach(() => {
  cleanup()
  closePluginInstallRequest()
  $connection.set(null)
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  vi.restoreAllMocks()
})

describe('PluginInstallModal hybrid install (#100412)', () => {
  it('serves the desktop half from the unified package on a local backend instead of copying it', async () => {
    setConnection('local')

    await clickInstall(await openModal())

    expect(installAgentPlugin).toHaveBeenCalledTimes(1)
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect(discoverRuntimePlugins).toHaveBeenCalledTimes(1)
  })

  it('enables the opt-in unified half because the user ticked "Desktop UI"', async () => {
    setConnection('local')

    await clickInstall(await openModal())

    expect($pluginDecisions.get()['word-count']).toBe(true)
    expect(notify).toHaveBeenCalledWith(successToast('enabled from the agent plugin package'))
  })

  it('rescans again when the first rescan was dropped by the loader lock, then enables it', async () => {
    // discoverRuntimePlugins() is a no-op while another scan holds its lock
    // (and a watched root has no poll to catch up); the next round lands.
    setConnection('local')
    discoverRuntimePlugins.mockReset()
    discoverRuntimePlugins.mockResolvedValueOnce(undefined).mockImplementationOnce(async () => {
      publishUnifiedHalf()
    })

    await clickInstall(await openModal())

    expect(discoverRuntimePlugins).toHaveBeenCalledTimes(2)
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect($pluginDecisions.get()['word-count']).toBe(true)
    expect(notify).toHaveBeenCalledWith(successToast('enabled from the agent plugin package'))
  })

  it('leaves the unified half opt-in and says where to enable it when no scan publishes it', async () => {
    setConnection('local')
    discoverRuntimePlugins.mockImplementation(async () => undefined)

    await clickInstall(await openModal())

    expect(discoverRuntimePlugins).toHaveBeenCalledTimes(3)
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect($pluginDecisions.get()).toEqual({})
    expect(notify).toHaveBeenCalledWith(successToast('enable it under Settings'))
  })

  it('skips the copy when the agent install fails as "already installed" (the unified half is on disk)', async () => {
    setConnection('local')
    unifiedHalfOnDisk = true
    installAgentPlugin.mockImplementation(async () => ({ ok: false, error: 'Plugin word-count is already installed' }))

    await clickInstall(await openModal())

    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect($pluginDecisions.get()['word-count']).toBe(true)
  })

  it('falls back to copying when the local agent install fails and nothing landed under plugins/', async () => {
    // Otherwise a clone failure would leave the user with no desktop half at
    // all — a wrong guess must cost a duplicate, never the loss.
    setConnection('local')
    installAgentPlugin.mockImplementation(async () => ({ ok: false, error: 'clone failed' }))

    await clickInstall(await openModal())

    expect(installDesktopPlugin).toHaveBeenCalledTimes(1)
    expect($pluginDecisions.get()).toEqual({})
  })

  it('copies when the backend installed into a root this app does not scan', async () => {
    // e.g. a profile-scoped hermes home: the agent install succeeded but
    // plugins/<name>/desktop/plugin.js is not in the scanned root.
    setConnection('local')
    installAgentPlugin.mockImplementation(async () => ({ ok: true, pluginName: 'word-count' }))

    await clickInstall(await openModal())

    expect(installDesktopPlugin).toHaveBeenCalledTimes(1)
    expect(discoverRuntimePlugins).toHaveBeenCalledTimes(1)
  })

  it('copies a root-level plugin.js desktop half (the unified door does not serve that shape)', async () => {
    setConnection('local')
    unifiedHalfOnDisk = true
    probePluginRepo.mockImplementation(async () => ({ ...hybridProbe, desktopSourceSubdir: '.' }))

    const install = await openModal()

    expect(screen.queryByText(/Uses the desktop half inside the agent plugin package/)).toBeNull()

    await clickInstall(install)

    expect(installDesktopPlugin).toHaveBeenCalledTimes(1)
  })

  it('reports an enable failure instead of swallowing it', async () => {
    setConnection('local')
    discoverRuntimePlugins.mockImplementation(async () => {
      publishPlugin(
        { id: 'word-count', name: 'Word count', kind: 'disk', status: 'disabled', file: UNIFIED_ENTRY },
        {
          activate: () => {
            throw new Error('register exploded')
          },
          deactivate: () => undefined
        }
      )
    })

    await clickInstall(await openModal())

    expect(notify).not.toHaveBeenCalledWith(successToast('enabled from the agent plugin package'))
    expect(await screen.findByText(/register exploded/)).toBeTruthy()
    // The enable decision was persisted before activation threw; it is rolled back.
    expect($pluginDecisions.get()['word-count']).toBe(false)
  })

  it('keeps refreshing an existing standalone copy: it is the one the loader serves', async () => {
    // An install made before this fix left desktop-plugins/word-count/. The
    // unified half must not silently win over it (and once the scan-time
    // dedupe lands, that stale copy would be the only one served).
    setConnection('local')
    standaloneOnDisk = true

    const install = await openModal()

    expect(screen.queryByText(/Uses the desktop half inside the agent plugin package/)).toBeNull()
    expect(screen.getByText(/Installs into this app's local desktop-plugins folder/)).toBeTruthy()

    await clickInstall(install)

    expect(installDesktopPlugin).toHaveBeenCalledWith({ identifier: REPO, force: false })
    expect($pluginDecisions.get()).toEqual({})
    expect(readDir).not.toHaveBeenCalledWith(PLUGINS_ROOT)
  })

  it('Force-reinstall refreshes an existing standalone copy instead of leaving it stale', async () => {
    setConnection('local')
    standaloneOnDisk = true
    renderModal()

    await act(async () => {
      openPluginInstallRequest({ repo: REPO, force: true })
    })

    const install = await screen.findByRole('button', { name: 'Install' })

    await waitFor(() => expect(install.hasAttribute('disabled')).toBe(false))
    await clickInstall(install)

    expect(installDesktopPlugin).toHaveBeenCalledWith({ identifier: REPO, force: true })
  })

  it('reports a unified half that failed to load right away instead of waiting every round', async () => {
    setConnection('local')
    discoverRuntimePlugins.mockImplementation(async () => {
      publishPlugin({
        id: 'word-count',
        name: 'word-count',
        kind: 'disk',
        status: 'error',
        error: 'syntax exploded',
        file: UNIFIED_ENTRY
      })
    })

    await clickInstall(await openModal())

    expect(discoverRuntimePlugins).toHaveBeenCalledTimes(1)
    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect($pluginDecisions.get()).toEqual({})
    expect(notify).not.toHaveBeenCalledWith(successToast('agent plugin package'))
    expect(await screen.findByText(/syntax exploded/)).toBeTruthy()
  })

  it('never enables the loader\'s "<id>:disk-shadowed" row for a plugin that ships bundled', async () => {
    setConnection('local')
    discoverRuntimePlugins.mockImplementation(async () => {
      publishPlugin({
        id: 'word-count:disk-shadowed',
        name: 'Word count (stale disk copy)',
        description: 'Shadowed by the bundled "word-count" plugin — this folder is no longer used and can be deleted.',
        kind: 'disk',
        status: 'disabled',
        file: UNIFIED_ENTRY
      })
    })

    await clickInstall(await openModal())

    expect($pluginDecisions.get()).toEqual({})
    expect(await screen.findByText(/Shadowed by the bundled/)).toBeTruthy()
  })

  it('leaves an already-running unified plugin alone on an "already exists" retry', async () => {
    // The user enabled it earlier; reinstalling must neither reload it nor
    // let a failing reload flip its persisted decision to false.
    setConnection('local')
    unifiedHalfOnDisk = true
    installAgentPlugin.mockImplementation(async () => ({ ok: false, error: "Plugin 'word-count' already exists" }))
    $pluginDecisions.set({ 'word-count': true })

    const activate = vi.fn(() => {
      throw new Error('must not be called')
    })

    discoverRuntimePlugins.mockImplementation(async () => {
      publishPlugin(
        { id: 'word-count', name: 'Word count', kind: 'disk', status: 'loaded', file: UNIFIED_ENTRY },
        { activate, deactivate: () => undefined }
      )
    })

    await clickInstall(await openModal())

    expect(installDesktopPlugin).not.toHaveBeenCalled()
    expect(activate).not.toHaveBeenCalled()
    expect($pluginDecisions.get()['word-count']).toBe(true)
    expect(notify).toHaveBeenCalledWith(successToast('enabled from the agent plugin package'))
    expect(screen.queryByText(/must not be called/)).toBeNull()
  })

  it('tells the user up front that no separate desktop copy will be made', async () => {
    setConnection('local')

    await openModal()

    expect(screen.getByText(/Uses the desktop half inside the agent plugin package/)).toBeTruthy()
  })

  it('still copies the desktop half when the agent half targets a remote backend', async () => {
    setConnection('remote')

    await clickInstall(await openModal())

    expect(installAgentPlugin).toHaveBeenCalledTimes(1)
    expect(installDesktopPlugin).toHaveBeenCalledTimes(1)
    expect(readDir).not.toHaveBeenCalledWith(PLUGINS_ROOT)
    expect(screen.queryByText(/Uses the desktop half inside the agent plugin package/)).toBeNull()
  })

  it('copies when the user unticks the agent half on a local backend', async () => {
    setConnection('local')

    const install = await openModal()

    fireEvent.click(screen.getByRole('checkbox', { name: /Agent plugin/ }))

    expect(screen.queryByText(/Uses the desktop half inside the agent plugin package/)).toBeNull()

    await clickInstall(install)

    expect(installAgentPlugin).not.toHaveBeenCalled()
    expect(installDesktopPlugin).toHaveBeenCalledTimes(1)
  })
})

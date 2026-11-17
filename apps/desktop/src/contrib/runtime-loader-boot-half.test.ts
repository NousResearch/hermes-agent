import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import type * as HermesModule from '@/hermes'
import { type AgentPluginRow, type GatewayRequest, loadAgentPlugins } from '@/store/agent-plugins'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

import { $pluginDecisions, $pluginRecords, noteEnabledAgentPackages, setPluginEnabled } from './plugins-store'
import { discoverRuntimePlugins } from './runtime-loader'

// getStatus would supply the connected backend's hermes_home. The boot scan
// must keep using the Electron-local desktop-plugins root.
const getStatus = vi.fn(async () => ({ hermes_home: '/remote/box/.hermes' }))

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<typeof HermesModule>()),
  getStatus: () => getStatus()
}))

const desktopPluginsRoot = vi.fn<() => Promise<string>>()
const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const readFileText = vi.fn<(path: string) => Promise<{ text: string; truncated?: boolean }>>()
const watchPreviewFile = vi.fn<(path: string) => Promise<{ id: string }>>()
const stopPreviewFileWatch = vi.fn<(id: string) => Promise<boolean>>()

function enabledRow(over: Partial<AgentPluginRow> & Pick<AgentPluginRow, 'name'>): AgentPluginRow {
  return {
    description: '',
    source: 'git',
    status: 'enabled',
    version: '1.0.0',
    ...over
  }
}

function listRequest(rows: AgentPluginRow[]): GatewayRequest {
  return (async () => ({ plugins: rows })) as GatewayRequest
}

/** Inventory one opt-in unified half, then run the real list + boot scan. */
async function withUnifiedHalf(
  id: string,
  packageName: string,
  run: (register: ReturnType<typeof vi.fn>) => Promise<void>
): Promise<void> {
  desktopPluginsRoot.mockResolvedValue('/local/.hermes/desktop-plugins')
  const root = '/local/.hermes/desktop-plugins'
  const folder = `${root}/${id}`

  readDir.mockImplementation(async dir => {
    if (dir === root) {
      return { entries: [{ isDirectory: true, name: id, path: folder }] }
    }

    if (dir === folder) {
      return {
        entries: [
          { isDirectory: false, name: '.hermes-package.json', path: `${folder}/.hermes-package.json` },
          { isDirectory: false, name: 'plugin.js', path: `${folder}/plugin.js` }
        ]
      }
    }

    return { entries: [] }
  })

  const register = vi.fn()

  const token = `__bootHalfRegister_${id.replace(/[^A-Za-z0-9_]/g, '_')}`

  ;(globalThis as Record<string, unknown>)[token] = register
  readFileText.mockImplementation(async file =>
    file.endsWith('.hermes-package.json')
      ? {
          text: JSON.stringify({
            package: packageName,
            source: `/x/plugins/${packageName}/desktop`,
            sourceMtimeMs: 1
          })
        }
      : { text: `export default { id: ${JSON.stringify(id)}, register: globalThis.${token} }` }
  )
  watchPreviewFile.mockResolvedValue({ id: `w-${id}` })

  const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockImplementation(
    blob =>
      `data:text/javascript;base64,${Buffer.from((blob as unknown as { parts: string[] }).parts.join('')).toString('base64')}`
  )

  const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  const RealBlob = globalThis.Blob

  vi.stubGlobal(
    'Blob',
    class {
      parts: string[]

      constructor(parts: string[]) {
        this.parts = parts
      }
    }
  )

  try {
    await run(register)
  } finally {
    createObjectURL.mockRestore()
    revokeObjectURL.mockRestore()
    vi.stubGlobal('Blob', RealBlob)
    delete (globalThis as Record<string, unknown>)[token]
    readDir.mockResolvedValue({ entries: [] })
    await discoverRuntimePlugins()
    resetDesktopChoices()
    $connection.set(null)
    $activeGatewayProfile.set('default')
  }
}

function resetDesktopChoices(): void {
  $pluginDecisions.set({})
  noteEnabledAgentPackages([])

  try {
    window.localStorage.removeItem('hermes.desktop.pluginDecisions.v2')
  } catch {
    // The live atom is the copy pluginActive reads; a missing storage file is nonfatal.
  }
}

beforeEach(() => {
  desktopPluginsRoot.mockReset()
  readDir.mockReset()
  readFileText.mockReset()
  watchPreviewFile.mockReset()
  stopPreviewFileWatch.mockReset()
  stopPreviewFileWatch.mockResolvedValue(true)
  getStatus.mockClear()
  resetDesktopChoices()
  $connection.set({ mode: 'local' } as never)
  $activeGatewayProfile.set('default')
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    desktopPluginsRoot,
    readDir,
    readFileText,
    stopPreviewFileWatch,
    watchPreviewFile
  }
})

afterEach(() => {
  resetDesktopChoices()
  $connection.set(null)
  $activeGatewayProfile.set('default')
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe.sequential('boot catch-up for an enabled unified package', () => {
  const meter = enabledRow({
    has_desktop_half: true,
    install_dir: '/home/u/.hermes/plugins/hermes-subscription-meter',
    name: 'subscription-meter'
  })

  it('enables the desktop half when the manifest name differs from the install folder', async () => {
    // Electron stamps packageName from plugins/<folder>/, not the manifest name.
    // `subscription-meter` cloned as hermes-subscription-meter is the reported miss.
    await withUnifiedHalf('meter-folder', 'hermes-subscription-meter', async register => {
      await loadAgentPlugins(listRequest([meter]), 'default')
      await discoverRuntimePlugins()

      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get()['meter-folder']).toMatchObject({
        packageName: 'hermes-subscription-meter',
        status: 'loaded'
      })
    })
  })

  it('uses the same folder identity for a Windows install dir and an omitted launch profile', async () => {
    await withUnifiedHalf('meter-win', 'hermes-subscription-meter', async register => {
      await loadAgentPlugins(
        listRequest([
          enabledRow({
            has_desktop_half: true,
            install_dir: 'C:\\Users\\u\\.hermes\\plugins\\hermes-subscription-meter',
            name: 'subscription-meter'
          })
        ]),
        null
      )
      await discoverRuntimePlugins()

      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get()['meter-win']?.status).toBe('loaded')
    })
  })

  it('does not enable a local half from a remote gateway list', async () => {
    $connection.set({ mode: 'remote' } as never)

    await withUnifiedHalf('meter-remote', 'hermes-subscription-meter', async register => {
      await loadAgentPlugins(listRequest([meter]), 'default')
      await discoverRuntimePlugins()

      expect(register).not.toHaveBeenCalled()
      expect($pluginRecords.get()['meter-remote']?.status).toBe('disabled')
    })
  })

  it('does not enable a local half from another profile list', async () => {
    $activeGatewayProfile.set('default')

    await withUnifiedHalf('meter-other', 'hermes-subscription-meter', async register => {
      await loadAgentPlugins(listRequest([meter]), 'work')
      await discoverRuntimePlugins()

      expect(register).not.toHaveBeenCalled()
      expect($pluginRecords.get()['meter-other']?.status).toBe('disabled')
    })
  })

  it('does not enable a local half for an agent-only row that shares the folder name', async () => {
    await withUnifiedHalf('meter-agent', 'hermes-subscription-meter', async register => {
      await loadAgentPlugins(
        listRequest([
          enabledRow({
            has_desktop_half: false,
            install_dir: '/home/u/.hermes/plugins/hermes-subscription-meter',
            name: 'subscription-meter'
          })
        ]),
        'default'
      )
      await discoverRuntimePlugins()

      expect(register).not.toHaveBeenCalled()
      expect($pluginRecords.get()['meter-agent']?.status).toBe('disabled')
    })
  })

  it('does not let a later remote list replace a local catch-up', async () => {
    await withUnifiedHalf('meter-keep', 'hermes-subscription-meter', async register => {
      await loadAgentPlugins(listRequest([meter]), 'default')
      $connection.set({ mode: 'remote' } as never)
      await loadAgentPlugins(listRequest([]), 'default')
      $connection.set({ mode: 'local' } as never)
      await discoverRuntimePlugins()

      expect(register).toHaveBeenCalledTimes(1)
      expect($pluginRecords.get()['meter-keep']?.status).toBe('loaded')
    })
  })

  it('leaves the half off when Desktop was explicitly switched off', async () => {
    await withUnifiedHalf('meter-off', 'hermes-subscription-meter', async register => {
      await discoverRuntimePlugins()
      expect($pluginRecords.get()['meter-off']?.status).toBe('disabled')

      await setPluginEnabled('meter-off', false)
      await loadAgentPlugins(listRequest([meter]), 'default')

      expect(register).not.toHaveBeenCalled()
      expect($pluginRecords.get()['meter-off']?.status).toBe('disabled')
    })
  })
})

import { BrowserWindow, dialog, type IpcMain, type IpcMainInvokeEvent } from 'electron'

import { applyConnectionChange } from './connection-apply'
import {
  normalizeLauncherUrl,
  normalizeProfileLauncher,
  profileHasRemoteConnection,
  profileLauncher
} from './connection-config'
import { applyConnectionConfigAtomically } from './connection-config-apply'
import {
  clearProfileBackendConfirm,
  type ProfileBackendConfirmCopy,
  setProfileBackendConfirm
} from './profile-backend-confirm'
import { createProfileBackendGrants } from './profile-backend-grants'

// Main's half of the plugin SDK's host.setProfileBackend / clearProfileBackend.
// A desktop plugin runs one profile's backend on this computer (its launcher on
// a loopback port, for example in front of a sandbox) and routes the profile
// to it. The route is the profile's own `mode: 'remote'` entry marked
// `launcher: { plugin }`. Every plugin shares one renderer realm, so main cannot
// tell callers apart: each change is confirmed natively, naming the plugin and
// the profile, and only a loopback URL is accepted.
//
// `confirm-set` / `confirm-clear` ask first and mint a one-time grant, so the
// plugin can start (prepare) or stop (beforeApply) its backend between the
// user's answer and the route change without a second dialog.

type ConnectionConfig = Record<string, any>

export interface ProfileBackendIpcDeps {
  assertCanMutateRouting: () => void
  cancelAndWait: (scope: string) => Promise<void>
  encryptToken: (token: string) => unknown
  mainWindow: () => BrowserWindow | null
  primaryProfile: () => string
  profileExists: (profile: string) => boolean
  readConfig: () => ConnectionConfig
  /** Tell the window that `profile`'s route moved (and only that profile's). */
  sendApplied: (detail: { profile: string }) => void
  stopPool: (scope: string) => unknown
  teardownPrimary: () => Promise<void>
  teardownSsh: (scope: null | string) => Promise<void>
  writeConfig: (config: ConnectionConfig) => void
}

interface Target {
  plugin: string
  profile: string
}

function profileBackendTarget(payload, profileExists: (profile: string) => boolean): Target {
  const profile = String(payload?.profile ?? '').trim() || 'default'
  const launcher = normalizeProfileLauncher({ plugin: payload?.plugin })

  if (profile !== 'default' && !profileExists(profile)) {
    throw new Error(`Profile "${profile}" does not exist.`)
  }

  if (!launcher) {
    throw new Error('A profile backend needs the id of the plugin that runs it.')
  }

  return { plugin: launcher.plugin, profile }
}

export function registerProfileBackendIpc(ipcMain: IpcMain, deps: ProfileBackendIpcDeps): void {
  const grants = createProfileBackendGrants()
  const target = payload => profileBackendTarget(payload, deps.profileExists)

  const confirm = async (event: IpcMainInvokeEvent, copy: ProfileBackendConfirmCopy): Promise<boolean> => {
    const parent = BrowserWindow.fromWebContents(event.sender) ?? deps.mainWindow()

    const options = {
      buttons: [...copy.buttons],
      cancelId: 1,
      defaultId: 1,
      detail: copy.detail,
      message: copy.message,
      noLink: true,
      type: 'question' as const
    }

    const { response } =
      parent && !parent.isDestroyed() ? await dialog.showMessageBox(parent, options) : await dialog.showMessageBox(options)

    return response === 0
  }

  // A plugin may route a profile no other plugin routes and the user has not
  // connected to a remote host.
  const assertSettable = ({ plugin, profile }: Target) => {
    const config = deps.readConfig()
    const current = profileLauncher(config, profile)

    if (current && current.plugin !== plugin) {
      throw new Error(`Profile "${profile}" already runs through the ${current.plugin} plugin.`)
    }

    if (!current && profileHasRemoteConnection(config, profile)) {
      throw new Error(`Profile "${profile}" already connects to a remote host.`)
    }
  }

  // The profile the calling plugin routes now, or null when no plugin routes it.
  const clearable = (payload): null | Target => {
    const { plugin, profile } = target(payload)
    const current = profileLauncher(deps.readConfig(), profile)

    if (current && current.plugin !== plugin) {
      throw new Error(`Profile "${profile}" runs through the ${current.plugin} plugin.`)
    }

    return current ? { plugin, profile } : null
  }

  // Write the profile's entry and retire whatever served it, with the same
  // atomic write and teardown a per-profile Settings apply uses. The route
  // lives only in connection.json: the machine's saved-gateways list (shared
  // with the Webapp) is neither rewritten nor restored, so a switch leaves its
  // revision alone and a failed one cannot drop a gateway saved meanwhile.
  const applyEntry = async (profile: string, entry: ConnectionConfig | null) => {
    const previousConfig = deps.readConfig()
    const { [profile]: _previous, ...others } = previousConfig.profiles || {}
    const profiles = entry ? { ...others, [profile]: entry } : others

    await applyConnectionConfigAtomically({
      previousConfig,
      previousRegistry: null,
      nextConfig: { ...previousConfig, profiles },
      nextRegistry: null,
      writeConfig: deps.writeConfig,
      writeRegistry: () => undefined,
      apply: () =>
        applyConnectionChange({
          cancelAndWait: deps.cancelAndWait,
          isPrimary: profile === deps.primaryProfile(),
          scope: profile,
          sendApplied: () => undefined,
          stopPool: deps.stopPool,
          teardownPrimary: deps.teardownPrimary,
          teardownSsh: deps.teardownSsh
        })
    })

    deps.sendApplied({ profile })
  }

  ipcMain.handle('hermes:profile-backend:confirm-set', async (event, payload) => {
    deps.assertCanMutateRouting()
    const { plugin, profile } = target(payload)
    assertSettable({ plugin, profile })

    if (!(await confirm(event, setProfileBackendConfirm({ plugin, profile, purpose: payload?.purpose })))) {
      return { confirmed: false }
    }

    return { confirmed: true, grant: grants.mint('set', plugin, profile) }
  })

  ipcMain.handle('hermes:profile-backend:set', async (event, payload) => {
    deps.assertCanMutateRouting()
    const { plugin, profile } = target(payload)
    const url = normalizeLauncherUrl(payload?.url)
    const token = String(payload?.token ?? '').trim()

    if (!token) {
      throw new Error('A profile backend needs its access token.')
    }

    assertSettable({ plugin, profile })

    // A grant from confirm-set stands for the user's answer; anything else asks now.
    const confirmed =
      grants.redeem(payload?.grant, 'set', plugin, profile) ||
      (await confirm(event, setProfileBackendConfirm({ plugin, profile, purpose: payload?.purpose, url })))

    if (!confirmed) {
      return { applied: false }
    }

    // The launcher keeps its own token in host-only storage, so a computer
    // without a keyring may store this loopback token as plain text too.
    const stored = deps.encryptToken(token)
    await applyEntry(profile, { mode: 'remote', url, authMode: 'token', token: stored, launcher: { plugin } })

    return { applied: true }
  })

  ipcMain.handle('hermes:profile-backend:confirm-clear', async (event, payload) => {
    deps.assertCanMutateRouting()
    const current = clearable(payload)

    if (!current || !(await confirm(event, clearProfileBackendConfirm(current)))) {
      return { confirmed: false }
    }

    return { confirmed: true, grant: grants.mint('clear', current.plugin, current.profile) }
  })

  ipcMain.handle('hermes:profile-backend:clear', async (event, payload) => {
    deps.assertCanMutateRouting()
    const current = clearable(payload)

    if (!current) {
      return { applied: false }
    }

    // A grant from confirm-clear stands for the user's answer; anything else asks now.
    const confirmed =
      grants.redeem(payload?.grant, 'clear', current.plugin, current.profile) ||
      (await confirm(event, clearProfileBackendConfirm(current)))

    if (!confirmed) {
      return { applied: false }
    }

    await applyEntry(current.profile, null)

    return { applied: true }
  })
}

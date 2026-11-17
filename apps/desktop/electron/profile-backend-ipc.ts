import { isDeepStrictEqual } from 'node:util'

import { BrowserWindow, dialog, type IpcMain, type IpcMainInvokeEvent } from 'electron'

import { applyConnectionChange } from './connection-apply'
import {
  localProfileEntry,
  normalizeLauncherUrl,
  normalizeProfileLauncher,
  profileHasRemoteConnection,
  profileLauncher
} from './connection-config'
import { backendScopeKey } from './connection-registry'
import { primaryDialClaimKey } from './primary-boot-join'
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
  /** Drop the dial in flight under this claim key; its callers are told to ask again (BackendDialClaims.forget). */
  forgetDial: (claimKey: string) => void
  mainWindow: () => BrowserWindow | null
  primaryProfile: () => string
  profileExists: (profile: string) => boolean
  readConfig: () => ConnectionConfig
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

  const entryOf = (profile: string): ConnectionConfig | null => deps.readConfig().profiles?.[profile] ?? null

  // What main still holds for the route a write replaced: the profile's pooled
  // descriptor and any dial in flight for it. A launcher is gone seconds before
  // its route is cleared, and a window that lost its socket asks main for the
  // profile's backend in that gap: that dial waits on the launcher's address for
  // main's whole readiness limit, and a dial in flight answers every later one
  // of its profile. Left alone it answers the dials the windows make once they
  // are told the profile moved, and the profile has no backend for minutes.
  const retireDials = (profile: string) => {
    const claimKey = backendScopeKey(null, profile)

    deps.stopPool(profile)
    deps.forgetDial(claimKey)
    deps.forgetDial(primaryDialClaimKey(claimKey, true))
  }

  // Replace one profile's entry in connection.json as it is now; every other
  // entry stays. In the same tick the old route's dials are retired, so every
  // dial from here on reads the entry just written.
  const writeEntry = (profile: string, entry: ConnectionConfig | null) => {
    const config = deps.readConfig()
    const { [profile]: _previous, ...others } = config.profiles || {}

    deps.writeConfig({ ...config, profiles: entry ? { ...others, [profile]: entry } : others })
    retireDials(profile)
  }

  // Every window dials the profile on its own socket, and a plugin's page runs
  // in any of them: each one is told that the profile's route moved (and only
  // that profile's), not the main window alone.
  const sendApplied = (profile: string) => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed() && !win.webContents.isDestroyed()) {
        win.webContents.send('hermes:connection:applied', { profile })
      }
    }
  }

  // Write the profile's entry and retire whatever served it, with the teardown
  // a per-profile Settings apply uses. The route lives only in connection.json:
  // the machine's saved-gateways list (shared with the Webapp) is neither
  // rewritten nor restored, so a switch leaves its revision alone and a failed
  // one cannot drop a gateway saved meanwhile.
  const applyEntry = async (profile: string, entry: ConnectionConfig | null) => {
    const previous = entryOf(profile)

    writeEntry(profile, entry)
    const written = entryOf(profile)

    try {
      await applyConnectionChange({
        cancelAndWait: deps.cancelAndWait,
        isPrimary: profile === deps.primaryProfile(),
        scope: profile,
        sendApplied: () => undefined,
        // Done with the write (retireDials): a descriptor pooled since then is the new route's.
        stopPool: () => undefined,
        teardownPrimary: deps.teardownPrimary,
        teardownSsh: deps.teardownSsh
      })
    } catch (error) {
      // The teardown failed. Put back this profile's entry alone, and only
      // while it is still the one written here: another profile's switch, or a
      // newer route for this one, that landed during the teardown stays.
      try {
        if (isDeepStrictEqual(entryOf(profile), written)) {
          writeEntry(profile, previous)
        }
      } catch {
        // Keep the teardown's error.
      }

      throw error
    }

    sendApplied(profile)
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
    if (!grants.redeem(payload?.grant, 'set', plugin, profile)) {
      if (!(await confirm(event, setProfileBackendConfirm({ plugin, profile, purpose: payload?.purpose, url })))) {
        return { applied: false }
      }

      // The dialog can stay open for minutes: who routes the profile is checked again.
      assertSettable({ plugin, profile })
    }

    // The launcher keeps its own token in host-only storage, so a computer
    // without a keyring may store this loopback token as plain text too.
    const stored = deps.encryptToken(token)
    // A profile's own local entry holds its saved SSH settings; they ride on the launcher entry until the clear.
    const kept = localProfileEntry(entryOf(profile))

    await applyEntry(profile, {
      mode: 'remote',
      url,
      authMode: 'token',
      token: stored,
      launcher: { plugin },
      ...(kept ? { savedSsh: kept.savedSsh } : {})
    })

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
    let current = clearable(payload)

    if (!current) {
      return { applied: false }
    }

    // A grant from confirm-clear stands for the user's answer; anything else asks now.
    if (!grants.redeem(payload?.grant, 'clear', current.plugin, current.profile)) {
      if (!(await confirm(event, clearProfileBackendConfirm(current)))) {
        return { applied: false }
      }

      // The dialog can stay open for minutes: who routes the profile is checked again.
      current = clearable(payload)

      if (!current) {
        return { applied: false }
      }
    }

    // The local entry the switch on replaced comes back (its saved SSH settings); a profile that had none has none.
    await applyEntry(current.profile, localProfileEntry(entryOf(current.profile)))

    return { applied: true }
  })
}

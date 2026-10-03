import fs from 'node:fs'
import crypto from 'node:crypto'
import path from 'node:path'
import { app, BrowserWindow, dialog, ipcMain } from 'electron'
import { PcDriverSession } from './pc-driver-session'
import { validateBrowserPrepare } from './browser-policy'
import { actionConsentScope, consentCovers, type ActionConsentScope } from './action-consent'
import { browserArguments } from './browser-session'
import { refreshConversationLifecycle, startBrowserLifecycle } from './browser-lifecycle'
import { OwnedBrowsers } from './owned-browser'
import { SafariBrowser } from './safari-browser'
import { ExistingBrowsers } from './existing-browser'
import { requestedProfileScope, needsProfileApproval } from './profile-access'
import { retainBrowserAfterError } from './browser-error-policy'

export const PC_TOOLS = new Set([
  'check_permissions', 'list_apps', 'list_windows', 'get_window_state', 'get_accessibility_tree',
  'get_desktop_state', 'click', 'double_click', 'right_click', 'drag', 'scroll', 'type_text',
  'press_key', 'hotkey', 'set_value', 'bring_to_front', 'launch_app', 'invoke_menu', 'verify_state',
  'browser_prepare', 'get_browser_state', 'browser_navigate', 'browser_click', 'browser_type', 'browser_pointer', 'browser_dialog'
])
const READS = new Set(['check_permissions', 'list_apps', 'list_windows', 'get_window_state', 'get_accessibility_tree', 'get_desktop_state', 'verify_state', 'get_browser_state'])

export function registerPcIpc(connectionScope: (senderId: number) => string) {
  const sessions = new Map<string, { driver: PcDriverSession; owned: OwnedBrowsers; safari: SafariBrowser; existing: ExistingBrowsers; profileGrants: Set<string>; lastUsed: number; grants: Set<ActionConsentScope>; browserActive: boolean; lifecycleError?: string }>()
  // Serialize consent and calls: no concurrent dialogs or stale-snapshot races.
  let queue: Promise<unknown> = Promise.resolve()
  let quitting = false
  const stop = (key: string) => { const entry = sessions.get(key); entry?.driver.close(); entry?.existing.close(); if (entry) { void entry.owned.close(); void entry.safari.close() }; sessions.delete(key) }
  const sweep = setInterval(() => {
    for (const [key, entry] of sessions) if (!entry.grants.size && !entry.profileGrants.size && Date.now() - entry.lastUsed > 5 * 60 * 1000) stop(key)
  }, 30000)
  sweep.unref()
  let maintenancePending = false
  const keepAlive = setInterval(() => {
    if (maintenancePending) return
    maintenancePending = true
    queue = queue.then(async () => {
      for (const [key, entry] of sessions) {
        if (!entry.grants.size && !entry.profileGrants.size) continue
        try {
          await entry.existing.refresh()
          const named = entry.browserActive ? String(browserArguments('get_browser_state', {}, key).session) : undefined
          if (!await refreshConversationLifecycle(entry.driver, named)) {
            entry.browserActive = false
            entry.lifecycleError = 'PC driver lifecycle expired; request fresh conversation access. No actions were replayed.'
          }
        } catch {
          entry.browserActive = false
          entry.lifecycleError = 'PC driver lifecycle check failed. No actions were replayed.'
        }
      }
    }).finally(() => { maintenancePending = false })
  }, 60000)
  keepAlive.unref()
  app.on('before-quit', event => {
    if (quitting) return
    quitting = true
    event.preventDefault()
    clearInterval(sweep); clearInterval(keepAlive)
    const closing = [...sessions.values()].map(entry => { entry.driver.close(); entry.existing.close(); return Promise.allSettled([entry.owned.close(), entry.safari.close()]) })
    sessions.clear()
    void Promise.race([Promise.allSettled(closing), new Promise(resolve => setTimeout(resolve, 5000))]).finally(() => app.quit())
  })

  ipcMain.handle('hermes:desktop:pc', (event, payload: unknown) => {
    const deadline = Date.now() + 120000
    const run = async () => {
      if (quitting || event.senderFrame !== event.sender.mainFrame || Date.now() > deadline) throw new Error('PC request is no longer valid.')
      const p = payload as { sessionId?: unknown; action?: unknown; arguments?: unknown }
      if (!p || typeof p.sessionId !== 'string' || !p.sessionId || p.sessionId.length > 256) throw new Error('A scoped conversation is required.')
      const gatewayScope = crypto.createHash('sha256').update(connectionScope(event.sender.id)).digest('hex')
      const key = `${event.sender.id}:${gatewayScope}:${p.sessionId}`
      if (p.action === 'revoke') { stop(key); return { content: [{ type: 'text', text: 'PC access revoked for this conversation.' }] } }
      const command = process.env.HERMES_DESKTOP_CUA_DRIVER
      if (p.action === 'status') return { content: [{ type: 'text', text: JSON.stringify({ platform: process.platform, installed: Boolean(command && fs.existsSync(command)), active: sessions.has(key), conversation_grants: [...(sessions.get(key)?.grants || [])], browser_profile_grants: [...(sessions.get(key)?.profileGrants || [])], browser_active: sessions.get(key)?.browserActive || sessions.get(key)?.owned.active || sessions.get(key)?.safari.active || sessions.get(key)?.existing.active || false, browser_lifecycle_error: sessions.get(key)?.lifecycleError || null }) }] }
      if (typeof p.action !== 'string' || !PC_TOOLS.has(p.action)) throw new Error('Unsupported PC-control action.')
      if (!p.arguments || typeof p.arguments !== 'object' || Array.isArray(p.arguments)) throw new Error('PC arguments must be an object.')
      const args = p.arguments as Record<string, unknown>
      // Transport owns session identity. Never accept caller-selected lifecycle scopes or local output paths.
      if ('session' in args || 'screenshot_out_file' in args) throw new Error('Session and screenshot output paths are managed by Desktop.')
      if (p.action === 'browser_prepare') {
        validateBrowserPrepare(args)
      }
      if (JSON.stringify(args).length > 64000) throw new Error('PC arguments exceed the 64 KB limit.')
      if (!command || !fs.existsSync(command)) throw new Error('The local PC driver is not installed.')
      const parent = BrowserWindow.fromWebContents(event.sender)
      if (!parent || event.sender.isDestroyed()) throw new Error('The owning Desktop window is unavailable.')
      if (!sessions.has(key)) {
        const answer = await dialog.showMessageBox(parent, {
          type: 'question', title: 'Allow this conversation to access your PC?',
          message: 'Allow Athena to inspect this device’s windows and screenshots?',
          detail: `Window text and screenshots are sent to your remote Hermes server and its configured AI/vision providers. They may contain private information.\nConversation: ${p.sessionId}\nActions offer Allow once or Allow for this conversation. Inspection-only access expires after five idle minutes. Revoke with desktop_pc or close this client.`,
          buttons: ['Cancel', 'Allow'], defaultId: 0, cancelId: 0
        })
        if (answer.response !== 1 || event.sender.isDestroyed() || Date.now() > deadline) throw new Error('PC access was not granted or the request expired.')
        const driver = new PcDriverSession(command)
        sessions.set(key, { driver, owned: new OwnedBrowsers(), safari: new SafariBrowser(), existing: new ExistingBrowsers(command, key), profileGrants: new Set(), lastUsed: Date.now(), grants: new Set(), browserActive: false })
        event.sender.once('destroyed', () => stop(key))
        try {
          await driver.ready
          const started = await driver.call('start_session', {})
          if (started?.isError) throw new Error('The freshly approved PC inspection lifecycle could not be started.')
        } catch (error) { stop(key); throw error }
      }
      const current = sessions.get(key)
      if (!READS.has(p.action) && (!current || !consentCovers(current.grants, p.action, args))) {
        const scope = actionConsentScope(p.action, args)
        const answer = await dialog.showMessageBox(parent, {
          type: 'question', title: 'Allow PC action?', message: `Athena wants to perform ${p.action} on this device.`,
          detail: `${JSON.stringify(args).slice(0, 4000)}\n\nAllow for this conversation covers ${scope === 'foreground' ? 'focus-changing actions' : 'background mouse, keyboard, and browser actions'} in this chat. It lasts until revoked or this client closes. Background approval does not cover focus-changing actions.`,
          buttons: ['Cancel', 'Allow once', 'Allow for this conversation'], defaultId: 0, cancelId: 0
        })
        if (![1, 2].includes(answer.response) || event.sender.isDestroyed() || Date.now() > deadline) throw new Error('PC action was not granted or the request expired.')
        if (answer.response === 2) sessions.get(key)?.grants.add(scope)
      }
      const entry = sessions.get(key)
      if (!entry) throw new Error('PC access expired. Request fresh state.')
      const mode = (args.profile as { mode?: string } | undefined)?.mode
      const selectedBrowser = String(args.browser || 'chrome') as 'chrome' | 'edge' | 'firefox'
      const profileScope = requestedProfileScope(p.action, args, entry.owned.accountScope(args.target_id) || entry.existing.scope(args))
      if (profileScope && needsProfileApproval(entry.profileGrants, profileScope)) {
        const existing = profileScope.startsWith('existing_profile:')
        const answer = await dialog.showMessageBox(parent, {
          type: 'question', title: 'Allow signed-in browser access?',
          message: existing ? 'Allow Athena to control this existing browser window?' : 'Allow Athena to use its persistent signed-in browser profile?',
          detail: `${profileScope}\nThis grants access to signed-in websites and browser data visible in the selected browser. Reads and screenshots go to your Hermes server and AI providers. ${existing ? 'Attachment setup may foreground the exact selected window and enable its browser control endpoint. Your everyday browser will not be terminated.' : 'This is a separate Athena profile retained on this device; it can sign in and sync with your account.'}\nPrivate-browser approval does not cover this mode.`,
          buttons: ['Cancel', 'Allow once', 'Allow for this conversation'], defaultId: 0, cancelId: 0
        })
        if (![1, 2].includes(answer.response) || event.sender.isDestroyed() || Date.now() > deadline) throw new Error('Signed-in browser access was not granted or expired.')
        if (answer.response === 2) entry.profileGrants.add(profileScope)
      }
      if (Date.now() > deadline) throw new Error('PC request expired before execution.')
      if (crypto.createHash('sha256').update(connectionScope(event.sender.id)).digest('hex') !== gatewayScope) throw new Error('Gateway changed before PC execution. Request fresh access.')
      entry.lastUsed = Date.now()
      try {
        if (p.action === 'browser_prepare' && args.browser === 'safari') return await entry.safari.prepare()
        if (typeof args.target_id === 'string' && args.target_id.startsWith('saf-')) return await entry.safari.call(p.action, args)
        if (p.action === 'browser_prepare' && mode === 'existing_profile') return await entry.existing.prepare(args)
        if (p.action !== 'browser_prepare' && entry.existing.handles(args)) return await entry.existing.call(p.action, args)
        if (p.action === 'browser_prepare' && mode === 'athena_profile') return await entry.owned.prepare(selectedBrowser, '', false, path.join(app.getPath('userData'), 'athena-browser-profiles', gatewayScope, selectedBrowser))
        if (p.action === 'browser_prepare' && (['linux', 'darwin'].includes(process.platform) || ['edge', 'firefox'].includes(String(args.browser)))) return await entry.owned.prepare(selectedBrowser)
        if (typeof args.target_id === 'string' && args.target_id.startsWith('obt-')) return await entry.owned.call(p.action, args)
        const nativeArgs = { ...args }
        if (p.action === 'browser_prepare') delete nativeArgs.browser
        const driverArgs = browserArguments(p.action, nativeArgs, key)
        if (p.action === 'browser_prepare') await startBrowserLifecycle(entry.driver, String(driverArgs.session))
        const result = await entry.driver.call(p.action, driverArgs)
        if (p.action === 'browser_prepare' && !result?.isError) {
          entry.browserActive = true
          entry.lifecycleError = undefined
        }
        return result
      }
      catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        if (retainBrowserAfterError(p.action, args, message)) {
          entry.lifecycleError = `Browser command failed: ${message.slice(0, 500)}. No action was replayed. Inspect fresh state before another action.`
        } else stop(key)
        throw error
      }
    }
    const result = queue.then(run).catch(error => ({ isError: true, content: [{ type: 'text', text: String(error instanceof Error ? error.message : error) }] }))
    queue = result.then(() => undefined)
    return result
  })
}


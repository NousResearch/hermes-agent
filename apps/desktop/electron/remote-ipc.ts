import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { app, BrowserWindow, dialog, ipcMain } from 'electron'
import {
  deviceId,
  RemoteCommandReceiver,
  signRemoteCommand,
  validateRemoteCommand,
  type RemoteApproval,
  type RemoteCommand
} from './remote-authorization'

interface Identity {
  publicKey: string
  privateKey: string
}
interface Origin {
  id: string
  name: string
  publicKey: string
}

function identity(): Identity {
  const directory = path.join(app.getPath('userData'), 'cross-device')
  const filename = path.join(directory, 'identity.json')
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 })
  if (!fs.existsSync(filename)) {
    const pair = crypto.generateKeyPairSync('ed25519')
    const value = {
      publicKey: pair.publicKey.export({ type: 'spki', format: 'pem' }).toString(),
      privateKey: pair.privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()
    }
    try {
      fs.writeFileSync(filename, JSON.stringify(value), { flag: 'wx', mode: 0o600 })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
    }
  }
  const value = JSON.parse(fs.readFileSync(filename, 'utf8')) as Identity
  if (
    deviceId(value.publicKey) !==
    deviceId(crypto.createPublicKey(value.privateKey).export({ type: 'spki', format: 'pem' }).toString())
  )
    throw new Error('Cross-device identity is inconsistent.')
  return value
}

/** Enrollment and command approval stay in the native process; the model cannot sign. */
export function registerRemoteIpc(
  connectionScope: (senderId: number) => string,
  execute: (payload: Record<string, unknown>) => Promise<unknown>
) {
  const enrollments = new Map<
    number,
    { scope: string; token: string; receiver: RemoteCommandReceiver; origins: { id: string; name: string }[] }
  >()
  const conversationGrants = new Map<number, Set<string>>()
  const scopes = new Map<number, string>()
  const versions = new Map<number, number>()
  const invalidate = (sender: number) => {
    enrollments.get(sender)?.receiver.close()
    enrollments.delete(sender)
    conversationGrants.delete(sender)
    versions.set(sender, (versions.get(sender) ?? 0) + 1)
  }
  const scope = (sender: number) => crypto.createHash('sha256').update(connectionScope(sender)).digest('hex')
  ipcMain.handle('hermes:desktop:remote', async (event, payload) => {
    const parent = BrowserWindow.fromWebContents(event.sender)
    if (!parent || event.sender.isDestroyed()) throw new Error('The Desktop window is unavailable.')
    if (!payload || typeof payload.action !== 'string') throw new Error('Remote action is required.')
    const own = identity(),
      id = deviceId(own.publicKey)
    const descriptor = { id, name: os.hostname(), platform: process.platform, publicKey: own.publicKey }
    const args = payload.arguments || {}
    const currentScope = scope(event.sender.id)
    if (scopes.get(event.sender.id) !== currentScope) invalidate(event.sender.id)
    scopes.set(event.sender.id, currentScope)
    const version = versions.get(event.sender.id)
    const stillAuthorized = () =>
      !event.sender.isDestroyed() &&
      scope(event.sender.id) === currentScope &&
      versions.get(event.sender.id) === version
    if (payload.action === 'describe') return { ...descriptor, routingIdentityVersion: 1 }
    if (payload.action === 'routing_identity') {
      if (typeof payload.sessionId !== 'string' || !payload.sessionId ||
          typeof args.challenge !== 'string' || !/^[a-f0-9]{64}$/.test(args.challenge) ||
          typeof args.conversation !== 'string' || !args.conversation || args.conversation.length > 1024 ||
          typeof args.authorize !== 'boolean') throw new Error('Invalid routing identity request.')
      if (args.authorize) {
        const choice = await dialog.showMessageBox(parent, {
          type: 'question', title: 'Continue this conversation on this device?',
          message: `Allow this conversation to use terminal and file tools on ${descriptor.name}?`,
          detail: `Conversation: ${payload.sessionId}\nThis changes the conversation's local execution device. Future terminal and file requests will use this computer. Interrupted commands will not be replayed.`,
          buttons: ['Cancel', 'Allow on this device'], defaultId: 0, cancelId: 0
        })
        if (choice.response !== 1 || !stillAuthorized()) throw new Error('Desktop handoff authorization was refused or the gateway changed. Stop and wait for the user.')
      }
      if (!stillAuthorized()) throw new Error('Desktop connection changed during identity verification.')
      // Domain separated, fixed fields: this signature can never approve a command.
      const proof = ['hermes-desktop-route-v1', args.challenge, args.conversation,
        currentScope, id, descriptor.name, args.authorize]
      return { ...descriptor, gatewayScope: currentScope, authorized: args.authorize,
        signature: crypto.sign(null, Buffer.from(JSON.stringify(proof)), own.privateKey).toString('base64') }
    }
    if (payload.action === 'status') {
      const enrollment = enrollments.get(event.sender.id)
      return { ...descriptor, receiving: !!enrollment, origins: enrollment?.origins ?? [] }
    }
    if (payload.action === 'revoke') {
      // One identity is shared by all windows in this Desktop installation.
      // A Settings revoke must also retire access and pending dialogs in its peers.
      for (const sender of scopes.keys()) invalidate(sender)
      return { revoked: true, id }
    }
    if (payload.action === 'enroll_target') {
      const origins = args.origins as Origin[]
      if (!Array.isArray(origins) || !origins.length || origins.length > 32)
        throw new Error('Register an originating device first.')
      const keys = new Map<string, string>()
      for (const origin of origins) {
        if (typeof origin.name !== 'string' || origin.name.length > 256 || deviceId(origin.publicKey) !== origin.id)
          throw new Error('Invalid originating device identity.')
        keys.set(origin.id, origin.publicKey)
      }
      const choice = await dialog.showMessageBox(parent, {
        type: 'question',
        title: 'Allow remote commands on this device?',
        message: `Allow commands on ${descriptor.name} after approval on ${origins.map(origin => origin.name).join(', ')}?`,
        detail:
          'Command approvals will appear on the originating device and name this destination. Only these enrolled device keys can authorize commands. Access ends when this client closes, changes gateway, or revokes enrollment.',
        buttons: ['Cancel', 'Allow remote commands'],
        defaultId: 0,
        cancelId: 0
      })
      if (choice.response !== 1 || !stillAuthorized())
        throw new Error('Remote enrollment was refused, revoked, or the gateway changed.')
      const token = crypto.randomBytes(32).toString('hex')
      const enrollment = crypto.createHash('sha256').update(token).digest('hex')
      enrollments.get(event.sender.id)?.receiver.close()
      enrollments.set(event.sender.id, {
        scope: currentScope,
        token,
        receiver: new RemoteCommandReceiver(id, enrollment, keys),
        origins: origins.map(origin => ({ id: origin.id, name: origin.name }))
      })
      return { ...descriptor, token, enrollment }
    }
    if (payload.action === 'approve') {
      if (typeof payload.sessionId !== 'string' || !payload.sessionId)
        throw new Error('An originating conversation is required.')
      if (typeof args.targetName !== 'string' || args.targetName.length > 256)
        throw new Error('A destination name is required.')
      const command = { ...args.command, originDevice: id, conversation: payload.sessionId } as RemoteCommand
      // Validate all signed fields before displaying the exact command to the user.
      validateRemoteCommand(command)
      if (command.expiresAt <= Date.now() || command.expiresAt > Date.now() + 120000)
        throw new Error('The approval request expired.')
      const grantKey = JSON.stringify([currentScope, payload.sessionId, command.targetDevice, command.enrollment])
      if (conversationGrants.get(event.sender.id)?.has(grantKey)) return signRemoteCommand(command, own.privateKey)
      const choice = await dialog.showMessageBox(parent, {
        type: 'question',
        title: `Execute on ${args.targetName}?`,
        message: `${descriptor.name} → ${args.targetName}`,
        detail: `Conversation: ${payload.sessionId}\nDestination device: ${command.targetDevice}\nDirectory: ${command.cwd ?? 'destination home'}\nTimeout: ${command.timeout}s\n\n${command.command}\n\nAllow once approves this command. Allow for this conversation covers terminal and file commands on this enrolled destination until revoked or this client closes. It does not cover another device.`,
        buttons: ['Cancel', 'Allow once', 'Allow for this conversation'],
        defaultId: 0,
        cancelId: 0
      })
      if (![1, 2].includes(choice.response) || !stillAuthorized() || Date.now() >= command.expiresAt)
        throw new Error('Remote command approval was refused or expired.')
      if (choice.response === 2) {
        if (!conversationGrants.has(event.sender.id)) conversationGrants.set(event.sender.id, new Set())
        conversationGrants.get(event.sender.id)!.add(grantKey)
      }
      return signRemoteCommand(command, own.privateKey)
    }
    if (payload.action === 'execute') {
      const enrollment = enrollments.get(event.sender.id)
      if (
        !enrollment ||
        enrollment.scope !== currentScope ||
        typeof args.token !== 'string' ||
        args.token !== enrollment.token
      )
        throw new Error('This device is not enrolled for this gateway connection.')
      const command = enrollment.receiver.consume(args.approval as RemoteApproval)
      return execute({ command: command.command, cwd: command.cwd, timeout: command.timeout, shell: 'bash' })
    }
    throw new Error('Unsupported remote-device action.')
  })
}

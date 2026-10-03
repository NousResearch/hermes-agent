import { afterAll, expect, test, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import crypto from 'node:crypto'
const mocks = vi.hoisted(() => ({
  handlers: new Map<string, Function>(),
  root: '',
  responses: [] as number[],
  prompts: [] as any[],
  pending: null as Promise<{ response: number }> | null
}))
vi.mock('electron', () => ({
  app: { getPath: () => mocks.root },
  BrowserWindow: { fromWebContents: () => ({}) },
  ipcMain: { handle: (name: string, handler: Function) => mocks.handlers.set(name, handler) },
  dialog: {
    showMessageBox: async (_parent: unknown, options: any) => {
      mocks.prompts.push(options)
      if (mocks.pending) {
        const pending = mocks.pending
        mocks.pending = null
        return pending
      }
      return { response: mocks.responses.shift() ?? 0 }
    }
  }
}))
import { registerRemoteIpc } from './remote-ipc'
import { deviceId, signRemoteCommand, type RemoteCommand } from './remote-authorization'
mocks.root = fs.mkdtempSync(path.join(os.tmpdir(), 'athena-remote-test-'))
afterAll(() => fs.rmSync(mocks.root, { recursive: true, force: true }))

test('routing proofs survive sender restarts and handoff requires native consent', async () => {
  let gateway = 'route-gateway-one'
  let executions = 0
  registerRemoteIpc(() => gateway, async () => { executions++; return {} })
  const invoke = (sender: number, authorize: boolean, challenge = 'a'.repeat(64)) =>
    mocks.handlers.get('hermes:desktop:remote')!(
      { sender: { id: sender, isDestroyed: () => false } },
      { action: 'routing_identity', sessionId: 'visible-chat',
        arguments: { challenge, conversation: 'durable-chat', authorize } }
    )
  const first = await invoke(900, false)
  const restarted = await invoke(901, false, 'b'.repeat(64))
  expect(first.id).toBe(restarted.id)
  const proof = ['hermes-desktop-route-v1', 'b'.repeat(64), 'durable-chat',
    restarted.gatewayScope, restarted.id, restarted.name, false]
  expect(crypto.verify(null, Buffer.from(JSON.stringify(proof)), restarted.publicKey,
    Buffer.from(restarted.signature, 'base64'))).toBe(true)
  expect(crypto.verify(null, Buffer.from(JSON.stringify(proof)), restarted.publicKey,
    Buffer.from(first.signature, 'base64'))).toBe(false)
  mocks.responses.push(0)
  await expect(invoke(901, true)).rejects.toThrow('refused')
  mocks.responses.push(1)
  expect((await invoke(901, true)).authorized).toBe(true)
  let accept!: (choice: { response: number }) => void
  mocks.pending = new Promise(resolve => { accept = resolve })
  const pending = invoke(901, true)
  gateway = 'route-gateway-two'
  accept({ response: 1 })
  await expect(pending).rejects.toThrow('gateway changed')
  await expect(invoke(901, false, 'malformed')).rejects.toThrow('Invalid')
  expect(executions).toBe(0)
})

test('native enrollment, cancellation, conversation grants, replay refusal and revocation', async () => {
  const executions: any[] = []
  let gateway = 'gateway-one'
  registerRemoteIpc(
    () => gateway,
    async payload => {
      executions.push(payload)
      return { output: 'fixture', success: true }
    }
  )
  const invoke = (sender: number, action: string, args: any = {}, sessionId = 'chat-one') =>
    mocks.handlers.get('hermes:desktop:remote')!(
      { sender: { id: sender, isDestroyed: () => false } },
      { action, arguments: args, sessionId }
    )
  const own = await invoke(1, 'describe')
  const pair = crypto.generateKeyPairSync('ed25519')
  const publicKey = pair.publicKey.export({ type: 'spki', format: 'pem' }).toString()
  const privateKey = pair.privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()
  const origin = { id: deviceId(publicKey), name: 'origin-fixture', publicKey }
  mocks.responses.push(0)
  await expect(invoke(2, 'enroll_target', { origins: [origin] })).rejects.toThrow('refused')
  mocks.responses.push(1)
  const enrollment = await invoke(2, 'enroll_target', { origins: [origin] })
  const status = await invoke(2, 'status')
  expect(status.receiving).toBe(true)
  expect(status.origins).toEqual([{ id: origin.id, name: 'origin-fixture' }])
  expect(JSON.stringify(status)).not.toContain(enrollment.token)
  const command: RemoteCommand = {
    version: 1,
    requestId: crypto.randomUUID(),
    originDevice: origin.id,
    targetDevice: own.id,
    enrollment: enrollment.enrollment,
    conversation: 'chat-one',
    command: 'printf fixture',
    cwd: null,
    timeout: 10,
    expiresAt: Date.now() + 100000
  }
  const approval = signRemoteCommand(command, privateKey)
  await invoke(2, 'execute', { token: enrollment.token, approval })
  await expect(invoke(2, 'execute', { token: enrollment.token, approval })).rejects.toThrow()
  expect(executions).toHaveLength(1)
  const originCommand = { ...command, originDevice: own.id, requestId: crypto.randomUUID() }
  mocks.responses.push(0)
  await expect(invoke(1, 'approve', { command: originCommand, targetName: 'destination-fixture' })).rejects.toThrow(
    'refused'
  )
  mocks.responses.push(2)
  await invoke(1, 'approve', { command: originCommand, targetName: 'destination-fixture' })
  const prompts = mocks.prompts.length
  await invoke(1, 'approve', {
    command: { ...originCommand, requestId: crypto.randomUUID() },
    targetName: 'destination-fixture'
  })
  expect(mocks.prompts).toHaveLength(prompts)
  await expect(
    invoke(1, 'approve', {
      command: { ...originCommand, enrollment: 'b'.repeat(64) },
      targetName: 'destination-fixture'
    })
  ).rejects.toThrow('refused')
  await invoke(2, 'revoke')
  expect((await invoke(2, 'status')).receiving).toBe(false)
  await expect(invoke(2, 'execute', { token: enrollment.token, approval })).rejects.toThrow('not enrolled')
  gateway = 'gateway-two'
  await expect(invoke(1, 'approve', { command: originCommand, targetName: 'destination-fixture' })).rejects.toThrow(
    'refused'
  )
  expect(executions).toHaveLength(1)
})

test('revocation invalidates a pending enrollment dialog before it can grant access', async () => {
  registerRemoteIpc(
    () => 'fixture-gateway',
    async () => ({})
  )
  const invoke = (action: string, args: any = {}, senderId = 88) =>
    mocks.handlers.get('hermes:desktop:remote')!(
      { sender: { id: senderId, isDestroyed: () => false } },
      { action, arguments: args, sessionId: 'fixture-chat' }
    )
  const own = await invoke('describe')
  let accept!: (choice: { response: number }) => void
  mocks.pending = new Promise(resolve => {
    accept = resolve
  })
  const enrollment = invoke('enroll_target', {
    origins: [{ id: own.id, name: 'fixture-origin', publicKey: own.publicKey }]
  })
  await invoke('revoke', {}, 89)
  accept({ response: 1 })
  await expect(enrollment).rejects.toThrow('revoked')
  expect((await invoke('status')).receiving).toBe(false)
})

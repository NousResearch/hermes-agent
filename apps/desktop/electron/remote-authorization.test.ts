import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import { test } from 'node:test'
import { deviceId, RemoteCommandReceiver, signRemoteCommand, type RemoteCommand } from './remote-authorization'

function fixture() {
  const key = crypto.generateKeyPairSync('ed25519')
  const publicKey = key.publicKey.export({ type: 'spki', format: 'pem' }).toString()
  const privateKey = key.privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()
  const target = 'device-' + 'a'.repeat(64)
  const enrollment = 'b'.repeat(64)
  const command: RemoteCommand = { version: 1, requestId: crypto.randomUUID(), originDevice: deviceId(publicKey), targetDevice: target, enrollment,
    conversation: 'origin-chat', command: 'hostname', cwd: null, timeout: 10, expiresAt: Date.now() + 60000 }
  return { command, privateKey, receiver: new RemoteCommandReceiver(target, enrollment, new Map([[command.originDevice, publicKey]])) }
}

test('signed command is accepted once, before dispatch', () => {
  const f = fixture(), approval = signRemoteCommand(f.command, f.privateKey)
  assert.equal(f.receiver.consume(approval).command, 'hostname')
  assert.throws(() => f.receiver.consume(approval), /already dispatched/)
})

test('changing approved command, chat, or destination is refused', () => {
  for (const change of [{ command: 'rm example' }, { conversation: 'other-chat' }, { targetDevice: 'device-other' }]) {
    const f = fixture(), approval = signRemoteCommand(f.command, f.privateKey)
    approval.command = { ...approval.command, ...change }
    assert.throws(() => f.receiver.consume(approval), /signature|another device/)
  }
})

test('expired approvals and revoked enrollment cannot execute', () => {
  const f = fixture(), approval = signRemoteCommand(f.command, f.privateKey)
  assert.throws(() => f.receiver.consume(approval, f.command.expiresAt), /expired/)
  f.receiver.close()
  assert.throws(() => f.receiver.consume(approval), /enrollment has ended/)
})

test('signed approval cannot cross enrollments or use an unknown origin', () => {
  const f = fixture(), approval = signRemoteCommand(f.command, f.privateKey)
  const anotherEnrollment = new RemoteCommandReceiver(f.command.targetDevice, 'c'.repeat(64), new Map())
  assert.throws(() => anotherEnrollment.consume(approval), /another device enrollment/)
  const unknownOrigin = new RemoteCommandReceiver(f.command.targetDevice, f.command.enrollment, new Map())
  assert.throws(() => unknownOrigin.consume(approval), /not enrolled/)
})

import crypto from 'node:crypto'

export interface RemoteCommand {
  version: 1
  requestId: string
  originDevice: string
  targetDevice: string
  enrollment: string
  conversation: string
  command: string
  cwd: string | null
  timeout: number
  expiresAt: number
}

export interface RemoteApproval {
  command: RemoteCommand
  signature: string
}

export function deviceId(publicKey: string): string {
  const key = crypto.createPublicKey(publicKey).export({ type: 'spki', format: 'der' })
  return 'device-' + crypto.createHash('sha256').update(key).digest('hex')
}

function bytes(command: RemoteCommand): Buffer {
  if (command.version !== 1 || !/^[a-f0-9-]{36}$/.test(command.requestId)
    || !command.originDevice.startsWith('device-') || !command.targetDevice.startsWith('device-')
    || !/^[a-f0-9]{64}$/.test(command.enrollment)
    || !command.conversation || command.conversation.length > 256
    || !command.command || command.command.length > 64000
    || (command.cwd !== null && (typeof command.cwd !== 'string' || command.cwd.length > 4096))
    || !Number.isInteger(command.timeout) || command.timeout < 1 || command.timeout > 60
    || !Number.isSafeInteger(command.expiresAt)) throw new Error('Invalid remote command.')
  // Fixed fields and order: no caller-defined approval metadata can alter authority.
  return Buffer.from(JSON.stringify({ version: command.version, requestId: command.requestId,
    originDevice: command.originDevice, targetDevice: command.targetDevice,
    enrollment: command.enrollment,
    conversation: command.conversation, command: command.command, cwd: command.cwd,
    timeout: command.timeout, expiresAt: command.expiresAt }))
}

export function validateRemoteCommand(command: RemoteCommand): void { bytes(command) }

/** Called only after the originating Desktop's native approval dialog accepts. */
export function signRemoteCommand(command: RemoteCommand, privateKey: string): RemoteApproval {
  const key = crypto.createPrivateKey(privateKey)
  if (deviceId(crypto.createPublicKey(key).export({ type: 'spki', format: 'pem' }).toString()) !== command.originDevice) {
    throw new Error('Approval key does not belong to the originating device.')
  }
  return { command, signature: crypto.sign(null, bytes(command), key).toString('base64') }
}

/** One receiver per enrolled gateway connection; disconnect destroys its enrollment. */
export class RemoteCommandReceiver {
  private consumed = new Map<string, number>()
  private closed = false

  constructor(private targetDevice: string, private enrollment: string, private enrolledOrigins: Map<string, string>) {}

  consume(approval: RemoteApproval, now = Date.now()): RemoteCommand {
    if (this.closed) throw new Error('Remote-device enrollment has ended.')
    const command = approval.command
    const payload = bytes(command)
    if (command.targetDevice !== this.targetDevice) throw new Error('Approval targets another device.')
    if (command.enrollment !== this.enrollment) throw new Error('Approval belongs to another device enrollment.')
    if (command.expiresAt <= now || command.expiresAt > now + 120000) throw new Error('Remote approval expired or exceeds its lifetime.')
    const publicKey = this.enrolledOrigins.get(command.originDevice)
    if (!publicKey || deviceId(publicKey) !== command.originDevice) throw new Error('Origin device is not enrolled.')
    if (!crypto.verify(null, payload, publicKey, Buffer.from(approval.signature, 'base64'))) throw new Error('Remote approval signature is invalid.')
    for (const [id, expiry] of this.consumed) if (expiry <= now) this.consumed.delete(id)
    if (this.consumed.has(command.requestId)) throw new Error('Remote command was already dispatched; do not replay.')
    // Consume before dispatch. A failed or timed-out command never regains authority.
    this.consumed.set(command.requestId, command.expiresAt)
    return { ...command }
  }

  close() { this.closed = true; this.enrolledOrigins.clear(); this.consumed.clear() }
}

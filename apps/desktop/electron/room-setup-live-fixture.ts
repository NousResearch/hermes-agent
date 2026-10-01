/** Native setup acceptance with real daemons; AES stands in only for OS custody. */
import crypto from 'node:crypto'
import fs from 'node:fs'

import { nativeRoomClient } from './native-room-client'
import { roomSetupCoordinator } from './room-setup'
import { RoomSetupError, roomSetupStore } from './room-setup-store'

const config = JSON.parse(fs.readFileSync(0, 'utf8'))
const key = crypto.randomBytes(32)
const store = roomSetupStore({ directory: config.directory,
  encrypt: value => {
    const iv = crypto.randomBytes(12), cipher = crypto.createCipheriv('aes-256-gcm', key, iv)
    return Buffer.concat([iv, cipher.update(value), cipher.final(), cipher.getAuthTag()]).toString('base64')
  },
  decrypt: value => {
    const bytes = Buffer.from(value, 'base64'), cipher = crypto.createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12))
    cipher.setAuthTag(bytes.subarray(-16))
    return Buffer.concat([cipher.update(bytes.subarray(12, -16)), cipher.final()]).toString()
  }
})
const input = { home: { connectionId: 'home', profile: 'default' }, name: 'Two gateways', members: [
  { member_id: 'one', handle: 'home', connectionId: 'home', profile: 'default' },
  { member_id: 'two', handle: 'peer', connectionId: 'peer', profile: 'default' }
] }
const connect = route => nativeRoomClient(config[route.connectionId], route.profile)
const coordinator = roomSetupCoordinator({ store, connect })
try {
  await coordinator.create({ ...input, members: [{ ...input.members[0], profile: 'unconfigured' }, input.members[1]] }, () => undefined)
  throw new Error('Invalid roster unexpectedly succeeded')
} catch (error) {if (!(error instanceof RoomSetupError)) {throw error}}
if ((await store.list()).records.length) {throw new Error('Refused creation left a false cleanup obligation')}
const result = await coordinator.create(input, () => undefined)
if ((await store.list()).records.length) {throw new Error('Successful setup retained credential obligations')}
// A new viewer owns no setup socket; it observes and controls the durable room.
const viewer = await connect(input.home)
try {
  const sent = await viewer.request('groups.send', { room_id: result.room.room_id, event_id: crypto.randomUUID(),
    payload: { text: '@peer native setup proof', thread_id: 'test' } })
  if (!sent.accepted) {throw new Error('Send refused')}
  const deadline = Date.now() + 60000
  while (true) {
    const state = await viewer.request('groups.state', { room_id: result.room.room_id })
    if (state.driver_status?.counts?.settled >= 1) {break}
    if (Date.now() > deadline) {throw new Error('Peer turn did not settle')}
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  const ended = await viewer.request('groups.disband', { room_id: result.room.room_id, cancel_id: crypto.randomUUID() })
  if (!ended.tombstone?.disbanded_at) {throw new Error('Disband unconfirmed')}
} finally {viewer.close()}
let drop = true
const interrupted = roomSetupCoordinator({ store, connect: async route => {
  const client = await connect(route)
  if (route.connectionId !== 'peer') {return client}
  return { close: client.close, async request(method, params) {
    if (method === 'groups.peer.invite') {
      if (!drop) {throw new RoomSetupError('setup_connection_lost')}
      await client.request(method, params)
      drop = false
      throw new RoomSetupError('setup_connection_lost')
    }
    return client.request(method, params)
  } }
} })
try {await interrupted.create(input, () => undefined); throw new Error('Lost reply unexpectedly succeeded')}
catch (error) {if (!(error instanceof RoomSetupError)) {throw error}}
if (!(await store.list()).records.some(record => record.kind === 'peer' && !record.grant)) {throw new Error('Lost issuance obligation was forgotten')}
const replacement = roomSetupCoordinator({ store, connect: async route => nativeRoomClient(config.home, route.profile) })
if (!(await replacement.recover()).pending) {throw new Error('Replacement gateway incorrectly settled the original grant')}
const recovered = await coordinator.recover()
if (recovered.pending || (await store.list()).records.length) {throw new Error('Original issued grant was not recovered and revoked')}
process.stdout.write(JSON.stringify({ created: true, peerTurn: true, detachedViewer: true, disbanded: true,
  lostIssuanceRecovered: true, replacementRefused: true }))

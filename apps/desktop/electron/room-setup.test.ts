import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { roomSetupCoordinator } from './room-setup'
import { RoomSetupError, roomSetupStore } from './room-setup-store'
import type { SetupRecord } from './room-setup-store'

function encryption() {
  const key = crypto.randomBytes(32)
  return {
    encrypt: (value: string) => {
      const nonce = crypto.randomBytes(12), cipher = crypto.createCipheriv('aes-256-gcm', key, nonce)
      return Buffer.concat([nonce, cipher.update(value), cipher.final(), cipher.getAuthTag()]).toString('base64')
    },
    decrypt: (value: string) => {
      const bytes = Buffer.from(value, 'base64'), cipher = crypto.createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12))
      cipher.setAuthTag(bytes.subarray(-16))
      return Buffer.concat([cipher.update(bytes.subarray(12, -16)), cipher.final()]).toString()
    }
  }
}

test('real journal writes fail closed and do not acknowledge a transformed first value', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'room-custody-'))
  const codec = encryption()
  let enabled = false, calls = 0, transform = false
  const store = roomSetupStore({ directory, decrypt: codec.decrypt,
    encrypt: value => {if (!enabled) {throw new RoomSetupError('secure_storage_required')}; calls++; return codec.encrypt(transform ? value.replace('private-grant', 'changed-grant') : value)} })
  const record: SetupRecord = { id: crypto.randomUUID(), setupId: crypto.randomUUID(), roomId: 'room', kind: 'peer',
    route: { connectionId: 'target', profile: 'default' }, installationId: 'original', grant: 'private-grant' }
  try {
    await expect(store.put(record)).rejects.toMatchObject({ reason: 'secure_storage_required' })
    expect(calls).toBe(0)
    enabled = true
    await store.put(record)
    expect(await fs.readFile(path.join(directory, record.id + '.json'), 'utf8')).not.toContain(record.grant)
    const bad = { ...record, id: crypto.randomUUID() }
    await fs.writeFile(path.join(directory, bad.id + '.json'), 'corrupt')
    expect(await store.list()).toEqual({ records: [record], unreadable: [bad.id] })
    transform = true
    await expect(store.put(record)).rejects.toMatchObject({ reason: 'setup_journal_write_failed' })
    transform = false
    const blocked = { ...record, id: crypto.randomUUID() }
    await fs.mkdir(path.join(directory, blocked.id + '.json'))
    await expect(store.put(blocked)).rejects.toMatchObject({ reason: 'setup_journal_write_failed' })
  } finally {await fs.rm(directory, { recursive: true, force: true })}
})

test('a late issued grant is journaled and compensated only on the original socket', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'room-setup-'))
  const store = roomSetupStore({ directory, ...encryption() })
  const events: string[] = []
  let retired = false, replaced = false
  const catalog = { installation_id: 'target', persistent_process: true, text: true, attachments: false, catalog_digest: 'digest' }
  const home = { close() {}, async request(method: string, params?: any): Promise<any> {
    events.push('home:' + method)
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true, methods: ['groups.discard'], authority_gateway_id: 'home' }}
    if (method === 'groups.create') {return { room: { ...params, authority_gateway_id: 'home', authority_epoch: 1 } }}
    if (method === 'groups.disband') {return { tombstone: { disbanded_at: 1 } }}
    throw new Error('Unexpected home mutation')
  } }
  const original = { close() {}, async request(method: string, params?: any): Promise<any> {
    events.push('original:' + method)
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true, methods: ['groups.discard'], authority_gateway_id: 'target',
      server_time: Date.now() / 1000, features: ['peer_setup_recovery'], room_link: {
        enabled: true, authentication: 'proof-v2', endpoint: { available: true, url: 'https://target.invalid' }, catalog } }}
    if (method === 'groups.peer.invite') {retired = true; replaced = true; return { grant: 'private-grant', catalog, target_profile: 'default' }}
    if (method === 'groups.peer.revoke') {
      expect(params.grant).toBe('private-grant')
      expect((await store.list()).records.some(record => record.grant === params.grant)).toBe(true)
      return { revoked: true }
    }
    throw new Error('Unexpected peer mutation')
  } }
  const coordinator = roomSetupCoordinator({ store, connect: async route => {
    if (route.connectionId === 'home') {return home}
    if (replaced) {throw new Error('Replacement must not receive the secret')}
    return original
  } })
  try {
    await expect(coordinator.create({ home: { connectionId: 'home', profile: 'default' }, name: 'Room', members: [
      { member_id: 'one', handle: 'one', profile: 'default', connectionId: 'home' },
      { member_id: 'two', handle: 'two', profile: 'default', connectionId: 'target' }
    ] }, () => {if (retired) {throw new RoomSetupError('setup_document_retired')}})).rejects.toMatchObject({ reason: 'setup_document_retired' })
    expect(events).toContain('original:groups.peer.revoke')
    expect(events).not.toContain('home:groups.peer.register')
    expect(await store.list()).toEqual({ records: [], unreadable: [] })
  } finally {await fs.rm(directory, { recursive: true, force: true })}
})

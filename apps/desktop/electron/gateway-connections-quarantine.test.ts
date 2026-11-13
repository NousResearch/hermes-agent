import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { afterEach, test } from 'vitest'

import { REGISTRY_QUARANTINE_CAP } from './connection-registry'
import {
  createGatewayConnectionsStore,
  GATEWAY_CONNECTIONS_FILE,
  readLegacyUserDataRegistry
} from './gateway-connections-store'

// Migration is lossless: a store and a Desktop list that each hold quarantined entries keep every
// one of them, even past the normalizer's cap, through later writes on both sides.
const repo = path.resolve(import.meta.dirname, '../../..')
const python = process.env.HERMES_PYTHON || 'python3'
const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

const broken = (who: string, n: number) =>
  Array.from({ length: n }, (_, i) => ({
    reason: 'entry-missing-url',
    entry: { kind: 'remote', label: `${who} ${i}` }
  }))

test('every quarantined entry survives the import past the cap, and later writes keep them', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gw-quar-'))
  dirs.push(dir)
  const root = path.join(dir, 'hermes')
  const userData = path.join(dir, 'userData')
  fs.mkdirSync(root, { recursive: true })
  fs.mkdirSync(userData, { recursive: true })
  const file = path.join(root, GATEWAY_CONNECTIONS_FILE)
  fs.writeFileSync(file, JSON.stringify({ version: 2, revision: 3, connections: [], quarantined: broken('web', 15) }))
  fs.writeFileSync(
    path.join(userData, 'connections.json'),
    JSON.stringify({ version: 2, connections: [], quarantined: broken('desk', 15) })
  )

  const store = createGatewayConnectionsStore({
    root,
    userData,
    legacyRegistry: () => readLegacyUserDataRegistry(userData, {}),
    decrypt: (secret: any) => String(secret?.value || '')
  })

  const kept = () => {
    const doc = JSON.parse(fs.readFileSync(file, 'utf8'))

    return [...(doc.quarantined || []), ...(doc.quarantinedOverflow || [])].map((q: any) => q.entry.label).sort()
  }

  const all = [...broken('web', 15), ...broken('desk', 15)].map(q => q.entry.label).sort()
  assert.ok(all.length > REGISTRY_QUARANTINE_CAP)
  store.read()
  assert.deepEqual(kept(), all)
  assert.ok(JSON.parse(fs.readFileSync(file, 'utf8')).importNotices.some((n: any) => n.field === 'quarantinedOverflow'))

  store.write({ ...store.read(), launchMode: 'last-used' })
  assert.deepEqual(kept(), all)

  const env = { HERMES_HOME: root, HOME: dir, PATH: process.env.PATH, PYTHONPATH: repo }
  const script = 'from hermes_cli import gateway_connections as g\ng.set_launch_mode("primary")'
  await promisify(execFile)(python, ['-c', script], { env })
  assert.deepEqual(kept(), all)
})

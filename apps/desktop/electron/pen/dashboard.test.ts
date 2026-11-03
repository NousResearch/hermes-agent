import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { afterEach, beforeEach, test } from 'vitest'

import { setPenLibraryHome } from '../pen-host'

import { handleDashboardRequest, setPenDocumentSwitcher } from './dashboard'
import type { PenDocument } from './state'

let home = ''

function canvas(name: string): string {
  const file = path.join(home, 'pens', name, `${name}.pen`)

  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, '{}')

  return file
}

const docFor = (file: string): PenDocument => ({ docId: 'd1', fileURI: pathToFileURL(file).href, displayName: 'x' })

beforeEach(() => {
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'pen-dashboard-'))
  setPenLibraryHome(home)
})

afterEach(() => {
  fs.rmSync(home, { force: true, recursive: true })
})

test('the dashboard lists every library canvas as a draft with recents timestamps', async () => {
  const login = canvas('Login')

  canvas('Dashboard')

  const { handled, value } = await handleDashboardRequest(docFor(login), 'documents-list', undefined)
  const docs = value as Array<{ uri: string; kind: number; name: string; timestamps: { lastOpenedAt?: number } }>

  assert.equal(handled, true)
  assert.deepEqual(docs.map(d => d.name).sort(), ['Dashboard', 'Login'])
  assert.ok(docs.every(d => d.kind === 1 && typeof d.timestamps.lastOpenedAt === 'number'))
})

test("the editor's thumbnail lands as preview.png and comes back for the dashboard", async () => {
  const login = canvas('Login')
  const png = new Uint8Array([137, 80, 78, 71])

  await handleDashboardRequest(docFor(login), 'documents-save-preview', { image: png })
  assert.ok(fs.existsSync(path.join(path.dirname(login), 'preview.png')))

  const uri = pathToFileURL(login).href
  const { value } = await handleDashboardRequest(docFor(login), 'documents-previews', { uris: [uri] })

  assert.deepEqual([...(value as Record<string, Uint8Array>)[uri]], [...png])
})

test('open and new answer at once and hand the switch to the host', async () => {
  const login = canvas('Login')
  const other = canvas('Other')
  const switched: unknown[] = []

  setPenDocumentSwitcher(async target => {
    switched.push(target)
  })

  const open = await handleDashboardRequest(docFor(login), 'documents-open', { uri: pathToFileURL(other).href })
  const fresh = await handleDashboardRequest(docFor(login), 'documents-new', undefined)

  assert.deepEqual([open.value, fresh.value], [undefined, undefined])
  await new Promise(resolve => setImmediate(resolve))
  assert.deepEqual(switched, [{ path: other }, {}])
})

test('delete removes library canvases only; a stray path is not reported removed', async () => {
  const login = canvas('Login')
  const stray = path.join(home, 'elsewhere.pen')

  fs.writeFileSync(stray, '{}')

  const uris = [pathToFileURL(login).href, pathToFileURL(stray).href]
  const { value } = await handleDashboardRequest(docFor(login), 'documents-remove', { uris, removal: 1 })

  assert.deepEqual(value, [uris[0]])
  assert.equal(fs.existsSync(login), false)
  assert.equal(fs.existsSync(stray), true)
})

test('anything else is left for the storage handler', async () => {
  assert.equal((await handleDashboardRequest(docFor(canvas('A')), 'storage-load', undefined)).handled, false)
})

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

const ELECTRON_DIR = path.dirname(fileURLToPath(import.meta.url))
const read = (name: string) => fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')

test('selection context menu sends only Chromium selectionText to the renderer', () => {
  const source = read('main.ts')

  assert.match(source, /label: 'Read Aloud'/)
  assert.match(source, /webContents\.send\('hermes:selection-speech:read', params\.selectionText\)/)
  assert.doesNotMatch(source, /hermes:selection-speech:read[^\n]*innerText/)
  assert.doesNotMatch(source, /hermes:selection-speech:read[^\n]*textContent/)
})

test('selection context menu opens translate with only Chromium selectionText', () => {
  const source = read('main.ts')

  assert.match(source, /label: 'Translate…'/)
  assert.match(source, /webContents\.send\('hermes:selection-translate:open', params\.selectionText\)/)
  assert.doesNotMatch(source, /hermes:selection-translate:open[^\n]*innerText/)
  assert.doesNotMatch(source, /hermes:selection-translate:open[^\n]*textContent/)
})

test('preload exposes removable listeners for selection-only read aloud and translate', () => {
  const source = read('preload.ts')

  assert.match(source, /selectionSpeech:\s*{/)
  assert.match(source, /ipcRenderer\.on\('hermes:selection-speech:read', listener\)/)
  assert.match(source, /removeListener\('hermes:selection-speech:read', listener\)/)
  assert.match(source, /selectionTranslate:\s*{/)
  assert.match(source, /ipcRenderer\.on\('hermes:selection-translate:open', listener\)/)
  assert.match(source, /removeListener\('hermes:selection-translate:open', listener\)/)
})

test('selected text exposes native macOS Look Up without replacing the selection', () => {
  const source = read('main.ts')

  assert.match(source, /label: 'Look Up'/)
  assert.match(source, /webContents\.showDefinitionForSelection\(\)/)
  assert.match(source, /IS_MAC/)
  assert.match(source, /hasSelection/)
})

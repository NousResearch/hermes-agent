import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { closeDocument, openDocument } from './documents'
import { onPenEvent } from './state'

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pen-documents-'))

afterEach(() => {
  fs.rmSync(dir, { force: true, recursive: true })
})

test('a document announces its open and close with the same id', async () => {
  const file = path.join(dir, 'Draft.pen')

  fs.writeFileSync(file, '{}')

  const seen: Array<[string, string]> = []
  const offOpen = onPenEvent('open-document', doc => seen.push(['open', doc.docId]))
  const offClose = onPenEvent('close-document', doc => seen.push(['close', doc.docId]))

  const doc = await openDocument(file)

  // Re-opening the same file is not a second open.
  await openDocument(file)
  closeDocument(doc.docId)

  offOpen()
  offClose()

  assert.deepEqual(seen, [
    ['open', doc.docId],
    ['close', doc.docId]
  ])
})

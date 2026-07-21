import assert from 'node:assert/strict'

import { test } from 'vitest'

import { openLocalFile, type OpenLocalFileDeps } from './open-local-file'

interface Recorder {
  preview: string[]
  openPath: string[]
  reveal: string[]
  logs: string[]
}

function harness(opts: {
  platform: NodeJS.Platform
  previewResult?: string | null
  openPathResult?: string | Error
  revealThrows?: boolean
}): { rec: Recorder; deps: OpenLocalFileDeps } {
  const rec: Recorder = { preview: [], openPath: [], reveal: [], logs: [] }

  const deps: OpenLocalFileDeps = {
    platform: opts.platform,
    log: message => rec.logs.push(message),
    openWithMacPreview: async target => {
      rec.preview.push(target)

      return opts.previewResult ?? null
    },
    openPath: async target => {
      rec.openPath.push(target)

      if (opts.openPathResult instanceof Error) {
        throw opts.openPathResult
      }

      return opts.openPathResult ?? ''
    },
    showItemInFolder: target => {
      rec.reveal.push(target)

      if (opts.revealThrows) {
        throw new Error('reveal boom')
      }
    }
  }

  return { rec, deps }
}

test('macOS PDF opens via Preview and skips the default handler on success', async () => {
  const { rec, deps } = harness({ platform: 'darwin', previewResult: null })

  await openLocalFile('/tmp/report.pdf', deps)

  assert.deepEqual(rec.preview, ['/tmp/report.pdf'])
  assert.deepEqual(rec.openPath, [], 'default handler must not run when Preview succeeds')
  assert.deepEqual(rec.reveal, [])
})

test('macOS PDF falls back to the default handler when Preview fails', async () => {
  const { rec, deps } = harness({
    platform: 'darwin',
    previewResult: 'Preview did not launch',
    openPathResult: ''
  })

  await openLocalFile('/tmp/report.pdf', deps)

  assert.deepEqual(rec.preview, ['/tmp/report.pdf'])
  assert.deepEqual(rec.openPath, ['/tmp/report.pdf'])
  assert.deepEqual(rec.reveal, [], 'no reveal when the default handler succeeds')
  assert.ok(rec.logs.some(l => l.includes('Preview open failed')))
})

test('macOS PDF reveals in folder when both Preview and the default handler fail', async () => {
  const { rec, deps } = harness({
    platform: 'darwin',
    previewResult: 'Preview did not launch',
    openPathResult: 'Failed to open path'
  })

  await openLocalFile('/tmp/report.pdf', deps)

  assert.deepEqual(rec.preview, ['/tmp/report.pdf'])
  assert.deepEqual(rec.openPath, ['/tmp/report.pdf'])
  assert.deepEqual(rec.reveal, ['/tmp/report.pdf'])
})

test('macOS non-PDF uses the default handler and never invokes Preview', async () => {
  const { rec, deps } = harness({ platform: 'darwin', openPathResult: '' })

  await openLocalFile('/tmp/notes.txt', deps)

  assert.deepEqual(rec.preview, [], 'Preview bypass is PDF-only')
  assert.deepEqual(rec.openPath, ['/tmp/notes.txt'])
  assert.deepEqual(rec.reveal, [])
})

test('non-macOS PDF uses the default handler and never invokes Preview', async () => {
  const { rec, deps } = harness({ platform: 'linux', openPathResult: '' })

  await openLocalFile('/tmp/report.pdf', deps)

  assert.deepEqual(rec.preview, [], 'Preview bypass is macOS-only')
  assert.deepEqual(rec.openPath, ['/tmp/report.pdf'])
  assert.deepEqual(rec.reveal, [])
})

test('Preview bypass matches the .pdf extension case-insensitively', async () => {
  const { rec, deps } = harness({ platform: 'darwin', previewResult: null })

  await openLocalFile('/tmp/REPORT.PDF', deps)

  assert.deepEqual(rec.preview, ['/tmp/REPORT.PDF'])
  assert.deepEqual(rec.openPath, [])
})

test('a rejected openPath is logged and does not reveal (parity with prior behavior)', async () => {
  const { rec, deps } = harness({
    platform: 'linux',
    openPathResult: new Error('openPath blew up')
  })

  await openLocalFile('/tmp/notes.txt', deps)

  assert.deepEqual(rec.openPath, ['/tmp/notes.txt'])
  assert.deepEqual(rec.reveal, [], 'a thrown openPath must not fall through to reveal')
  assert.ok(rec.logs.some(l => l.includes('openPath rejected') && l.includes('openPath blew up')))
})

test('a throwing showItemInFolder is swallowed and logged', async () => {
  const { rec, deps } = harness({
    platform: 'linux',
    openPathResult: 'Failed to open path',
    revealThrows: true
  })

  await openLocalFile('/tmp/notes.txt', deps)

  assert.deepEqual(rec.reveal, ['/tmp/notes.txt'])
  assert.ok(rec.logs.some(l => l.includes('showItemInFolder failed')))
})

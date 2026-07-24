import assert from 'node:assert/strict'

import { beforeEach, test, vi } from 'vitest'

// resolveLocalReadPath is the WSL->Windows bridge every file-read boundary must
// route its raw path through on a Windows host. Spy on it so each test asserts
// the boundary *applies* the bridge (and with what argument) rather than
// re-testing the translation itself, which wsl-path-bridge.test.ts already covers.
const { bridge } = vi.hoisted(() => ({ bridge: vi.fn() }))

vi.mock('./wsl-path-bridge', () => ({ resolveLocalReadPath: bridge }))

import { resolveIpcFileReadPath, resolveMediaRequestPath, resolvePreviewTargetPath } from './local-read-path'

const BRIDGED = '\\\\wsl.localhost\\Ubuntu\\home\\alex\\file'

beforeEach(() => {
  bridge.mockReset()
  bridge.mockImplementation(() => BRIDGED)
})

test('resolveMediaRequestPath (hermes-media:// handler) bridges the decoded request pathname', () => {
  // Mirror how the renderer builds the URL: hermes-media://stream/<encoded path>.
  const url = new URL(`hermes-media://stream/${encodeURIComponent('/home/alex/My Clips/clip.mp4')}`)

  const result = resolveMediaRequestPath(url.pathname)

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/home/alex/My Clips/clip.mp4')
  assert.equal(result, BRIDGED)
})

test('resolveMediaRequestPath strips leading slashes before decoding and bridging', () => {
  resolveMediaRequestPath(`///${encodeURIComponent('/mnt/c/Users/alex/clip.mp4')}`)

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/mnt/c/Users/alex/clip.mp4')
})

test('resolveMediaRequestPath tolerates nullish pathname', () => {
  const result = resolveMediaRequestPath(undefined as unknown as string)

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '')
  assert.equal(result, BRIDGED)
})

test('resolveIpcFileReadPath (hermes:readFileDataUrl / hermes:readFileText) bridges the supplied path', () => {
  const result = resolveIpcFileReadPath('/home/alex/notes.txt')

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/home/alex/notes.txt')
  assert.equal(result, BRIDGED)
})

test('resolveIpcFileReadPath coerces a nullish path to an empty string before bridging', () => {
  resolveIpcFileReadPath(null)

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '')
})

test('resolvePreviewTargetPath (previewFileTarget) expands and bridges a plain backend target', () => {
  const expandUserPath = vi.fn((value: string) => value.replace('~', '/home/alex'))

  const result = resolvePreviewTargetPath('~/docs/readme.md', expandUserPath)

  assert.equal(expandUserPath.mock.calls.length, 1)
  assert.equal(expandUserPath.mock.calls[0][0], '~/docs/readme.md')
  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/home/alex/docs/readme.md')
  assert.equal(result, BRIDGED)
})

test('resolvePreviewTargetPath passes file: URLs through the bridge without expanding', () => {
  const expandUserPath = vi.fn((value: string) => value)

  resolvePreviewTargetPath('file:///home/alex/report.html', expandUserPath)

  assert.equal(expandUserPath.mock.calls.length, 0)
  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], 'file:///home/alex/report.html')
})

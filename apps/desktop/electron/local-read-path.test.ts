import assert from 'node:assert/strict'

import { beforeEach, test, vi } from 'vitest'

// resolveLocalReadPath is the WSL->Windows bridge every file-read boundary must
// route its raw path through on a Windows host. Spy on it so each test asserts
// the boundary *applies* the bridge (and with what argument) rather than
// re-testing the translation itself, which wsl-path-bridge.test.ts already covers.
const { bridge } = vi.hoisted(() => ({ bridge: vi.fn() }))

vi.mock('./wsl-path-bridge', () => ({ resolveLocalReadPath: bridge }))

import {
  resolveIpcFileReadPath,
  resolveMediaStreamFile,
  resolvePreviewTargetPath
} from './local-read-path'

const BRIDGED = '\\\\wsl.localhost\\Ubuntu\\home\\alex\\file'

beforeEach(() => {
  bridge.mockReset()
  bridge.mockImplementation(() => BRIDGED)
})

// Regression for hermes-agent 123823: the media protocol handler decodes the
// request pathname itself (parseMediaProtocolTarget in media-protocol.ts), so
// the resolveLocalFile boundary must NOT decode or strip leading slashes again.
// The pre-fix wiring ran decodeURIComponent + strip on the already-decoded
// path, turning `/home/...` into a cwd-relative `home/...` that ENOENTs into
// the handler's silent 404, and threw URIError on filenames with a literal `%`.
test('resolveMediaStreamFile (hermes-media:// resolveLocalFile) bridges the already-decoded path unchanged', () => {
  const result = resolveMediaStreamFile('/home/alex/My Clips/clip.mp4')

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/home/alex/My Clips/clip.mp4')
  assert.equal(result, BRIDGED)
})

test('resolveMediaStreamFile preserves a leading slash so absolute POSIX paths stay absolute', () => {
  // What the protocol handler hands over for hermes-media://stream/%2Fhome%2F...:
  // already decoded, still absolute. Stripping the leading slash here is the
  // 123823 regression (cwd-relative ENOENT -> silent 404 on Linux/macOS).
  const url = new URL(`hermes-media://stream/${encodeURIComponent('/home/alex/clip.mp4')}`)
  const filePath = decodeURIComponent(url.pathname.replace(/^\/+/, ''))

  resolveMediaStreamFile(filePath)

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/home/alex/clip.mp4')
})

test('resolveMediaStreamFile keeps percent-sign bytes in filenames intact', () => {
  // A filename containing a literal `%` (e.g. "100% clip.mp4") arrives
  // decoded; a second decodeURIComponent would throw URIError.
  resolveMediaStreamFile('/home/alex/100% clip.mp4')

  assert.equal(bridge.mock.calls.length, 1)
  assert.equal(bridge.mock.calls[0][0], '/home/alex/100% clip.mp4')
})

test('resolveMediaStreamFile tolerates nullish input', () => {
  const result = resolveMediaStreamFile(undefined as unknown as string)

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

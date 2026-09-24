import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  buildSideChatWindowUrl,
  normalizeSideChatAsk,
  normalizeSideChatContext,
  normalizeSideChatReply,
  sideChatWindowBounds
} from './side-chat'

test('sideChatWindowBounds pins the window to the right edge, vertically centered', () => {
  assert.deepEqual(sideChatWindowBounds({ x: 0, y: 25, width: 1440, height: 875 }), {
    x: 996,
    y: 153,
    width: 420,
    height: 620
  })
})

test('sideChatWindowBounds respects a work area that does not start at the origin', () => {
  const bounds = sideChatWindowBounds({ x: 1920, y: 0, width: 1280, height: 1024 })

  assert.equal(bounds.x, 1920 + 1280 - 420 - 24)
  assert.equal(bounds.y, Math.round((1024 - 620) / 2))
})

test('sideChatWindowBounds fits a work area smaller than the default window', () => {
  // The composer lives at the bottom of the window: an un-clamped height would
  // put it below the screen and make the whole feature un-typeable.
  assert.deepEqual(sideChatWindowBounds({ x: 0, y: 0, width: 400, height: 300 }), {
    x: 0,
    y: 0,
    width: 400,
    height: 300
  })
})

test('sideChatWindowBounds drops the edge margin rather than going off-screen', () => {
  // Work area exactly as wide as the window: insisting on the 24px gap would
  // push it past the right edge.
  assert.deepEqual(sideChatWindowBounds({ x: 0, y: 0, width: 420, height: 620 }), {
    x: 0,
    y: 0,
    width: 420,
    height: 620
  })
})

test('sideChatWindowBounds keeps a spawn fallback with no display', () => {
  assert.deepEqual(sideChatWindowBounds(), { x: 0, y: 0, width: 420, height: 620 })
})

test('buildSideChatWindowUrl puts ?win=side before the hash route', () => {
  // After the '#' HashRouter would read it as the route and the renderer would
  // never mount the side chat at all.
  assert.equal(
    buildSideChatWindowUrl({ devServer: 'http://127.0.0.1:5174' }),
    'http://127.0.0.1:5174/?win=side#/'
  )
  assert.equal(buildSideChatWindowUrl({ devServer: 'http://127.0.0.1:5174/' }), 'http://127.0.0.1:5174/?win=side#/')
})

test('buildSideChatWindowUrl loads the packaged renderer as a file URL', () => {
  const url = buildSideChatWindowUrl({ rendererIndexPath: '/opt/hermes/renderer/index.html' })

  assert.equal(url, 'file:///opt/hermes/renderer/index.html?win=side#/')
})

test('normalizeSideChatContext requires a parent session id', () => {
  assert.equal(normalizeSideChatContext(null), null)
  assert.equal(normalizeSideChatContext({ question: 'why?' }), null)
  assert.equal(normalizeSideChatContext({ sessionId: '   ' }), null)
})

test('normalizeSideChatContext keeps the question verbatim and trims the rest', () => {
  assert.deepEqual(normalizeSideChatContext({ sessionId: ' s1 ', title: ' Refactor ', question: 'which file?' }), {
    question: 'which file?',
    sessionId: 's1',
    title: 'Refactor'
  })
})

test('normalizeSideChatContext tolerates a bare /btw with no question', () => {
  assert.deepEqual(normalizeSideChatContext({ sessionId: 's1' }), { question: '', sessionId: 's1', title: '' })
})

test('normalizeSideChatAsk drops anything that could never be answered or shown', () => {
  assert.equal(normalizeSideChatAsk({ sessionId: 's1', text: 'hi' }), null, 'no askId to correlate the reply')
  assert.equal(normalizeSideChatAsk({ askId: 'a1', text: 'hi' }), null, 'no session to snapshot')
  assert.equal(normalizeSideChatAsk({ askId: 'a1', sessionId: 's1', text: '   ' }), null, 'blank question')
})

test('normalizeSideChatAsk passes a well-formed ask through', () => {
  assert.deepEqual(normalizeSideChatAsk({ askId: ' a1 ', sessionId: ' s1 ', text: 'which file was that?' }), {
    askId: 'a1',
    sessionId: 's1',
    text: 'which file was that?'
  })
})

test('normalizeSideChatReply keeps an empty answer when it carries an error', () => {
  // Otherwise a failed aside leaves its bubble spinning forever.
  assert.deepEqual(normalizeSideChatReply({ askId: 'a1', error: 'backend is retiring', text: '' }), {
    askId: 'a1',
    error: 'backend is retiring',
    text: ''
  })
})

test('normalizeSideChatReply drops a reply that is neither an answer nor a failure', () => {
  assert.equal(normalizeSideChatReply({ askId: 'a1', text: '   ' }), null)
  assert.equal(normalizeSideChatReply({ text: 'orphan answer' }), null)
})

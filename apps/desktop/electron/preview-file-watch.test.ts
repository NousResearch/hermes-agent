/**
 * Unit tests for owner-routed preview/plugin file-watch delivery (#108189).
 * Secondary windows filter by their own watch id, so events must land on the
 * registering WebContents — never the primary alone.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  PREVIEW_FILE_CHANGED_CHANNEL,
  sendPreviewFileChangedToOwner,
  type PreviewFileChangedPayload
} from './preview-file-watch'

function makeOwner() {
  const sends: Array<{ channel: string; payload: unknown }> = []
  let destroyed = false

  return {
    sends,
    isDestroyed: () => destroyed,
    destroy() {
      destroyed = true
    },
    send(channel: string, payload: unknown) {
      sends.push({ channel, payload })
    }
  }
}

const payload: PreviewFileChangedPayload = {
  id: 'watch-secondary',
  path: '/tmp/plugin.js',
  url: 'file:///tmp/plugin.js'
}

describe('sendPreviewFileChangedToOwner', () => {
  test('delivers to the owning WebContents, not a different window', () => {
    const owner = makeOwner()
    const other = makeOwner()

    assert.equal(sendPreviewFileChangedToOwner(owner, payload), true)
    assert.deepEqual(owner.sends, [{ channel: PREVIEW_FILE_CHANGED_CHANNEL, payload }])
    assert.deepEqual(other.sends, [])
  })

  test('tears the watch down when the owner is destroyed', () => {
    const owner = makeOwner()
    owner.destroy()
    let tornDown = false

    assert.equal(
      sendPreviewFileChangedToOwner(owner, payload, () => {
        tornDown = true
      }),
      false
    )
    assert.equal(tornDown, true)
    assert.deepEqual(owner.sends, [])
  })

  test('tears the watch down when the owner is missing', () => {
    let tornDown = false

    assert.equal(
      sendPreviewFileChangedToOwner(null, payload, () => {
        tornDown = true
      }),
      false
    )
    assert.equal(tornDown, true)
  })
})

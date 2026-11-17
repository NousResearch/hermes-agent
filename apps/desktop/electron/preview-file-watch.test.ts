/**
 * Unit tests for owner-routed preview/plugin file-watch delivery (#108189).
 * Secondary windows filter by their own watch id, so events must land on the
 * registering WebContents — never the primary alone.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  onPreviewWatchOwnerDestroyed,
  PREVIEW_FILE_CHANGED_CHANNEL,
  type PreviewFileChangedPayload,
  type PreviewWatchOwner,
  sendPreviewFileChangedToOwner
} from './preview-file-watch'

function makeOwner(): PreviewWatchOwner & {
  sends: Array<{ channel: string; payload: unknown }>
  destroy: () => void
  emitDestroyed: () => void
} {
  const sends: Array<{ channel: string; payload: unknown }> = []
  let destroyed = false
  const destroyedListeners = new Set<() => void>()

  return {
    sends,
    isDestroyed: () => destroyed,
    destroy() {
      destroyed = true
    },
    emitDestroyed() {
      for (const listener of destroyedListeners) {
        listener()
      }
    },
    once(event: 'destroyed', listener: () => void) {
      destroyedListeners.add(listener)

      return this
    },
    removeListener(event: 'destroyed', listener: () => void) {
      destroyedListeners.delete(listener)
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

describe('onPreviewWatchOwnerDestroyed', () => {
  test('runs the teardown the moment the owner window is destroyed', () => {
    const owner = makeOwner()
    let tornDown = 0

    onPreviewWatchOwnerDestroyed(owner, () => {
      tornDown += 1
    })

    // No teardown before the window dies, exactly one at destruction.
    assert.equal(tornDown, 0)
    owner.emitDestroyed()
    assert.equal(tornDown, 1)
  })

  test('an explicit watch close detaches the destroyed listener', () => {
    const owner = makeOwner()
    let tornDown = 0

    const off = onPreviewWatchOwnerDestroyed(owner, () => {
      tornDown += 1
    })

    off()
    owner.emitDestroyed()

    // The watch's own close() must not leave a stale teardown armed — and
    // must not resurrect a second one through the same listener.
    assert.equal(tornDown, 0)
  })

  test('tolerates a missing owner without exploding', () => {
    const off = onPreviewWatchOwnerDestroyed(null, () => {})

    assert.equal(off(), undefined)
  })
})

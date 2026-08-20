import assert from 'node:assert/strict'

import { beforeEach, test, vi } from 'vitest'

const registeredSessions: string[] = []
const requestHandlers: Array<(details: unknown, callback: (result: unknown) => void) => void> = []
let failingSession = ''

vi.mock('electron', () => ({
  session: {
    defaultSession: {
      webRequest: {
        onBeforeSendHeaders(handler) {
          if (failingSession === 'default') {
            throw new Error('default session registration failed')
          }

          registeredSessions.push('default')
          requestHandlers.push(handler)
        }
      }
    },
    fromPartition(partition: string) {
      return {
        webRequest: {
          onBeforeSendHeaders(handler) {
            if (failingSession === partition) {
              throw new Error('partition registration failed')
            }

            registeredSessions.push(partition)
            requestHandlers.push(handler)
          }
        }
      }
    }
  }
}))

import { installEmbedReferer } from './embed-referer'

beforeEach(() => {
  registeredSessions.length = 0
  requestHandlers.length = 0
  failingSession = ''
})

test('installEmbedReferer covers plain iframe requests in the default session', () => {
  installEmbedReferer()

  assert.deepEqual(registeredSessions, ['default', 'persist:hermes-embed'])
})

test('YouTube requests identify the Hermes app instead of impersonating YouTube', () => {
  installEmbedReferer()
  let result

  requestHandlers[0](
    { url: 'https://www.youtube.com/embed/video', requestHeaders: {} },
    value => {
      result = value
    }
  )

  assert.equal(result.requestHeaders.Referer, 'https://com.nousresearch.hermes/')
})

test.each([
  ['default', ['persist:hermes-embed']],
  ['persist:hermes-embed', ['default']]
])('a failed %s registration does not disable the other session', (failed, registered) => {
  failingSession = failed

  installEmbedReferer()

  assert.deepEqual(registeredSessions, registered)
})

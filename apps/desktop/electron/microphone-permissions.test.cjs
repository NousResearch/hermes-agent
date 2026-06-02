const assert = require('node:assert/strict')
const test = require('node:test')

const {
  MICROPHONE_PRIVACY_SETTINGS_URL,
  openMicrophonePrivacySettings,
  requestMicrophoneAccess
} = require('./microphone-permissions.cjs')

function shellMock() {
  const calls = []

  return {
    calls,
    shell: {
      openExternal: async url => {
        calls.push(url)
      }
    }
  }
}

test('non-mac platforms do not request microphone media access', async () => {
  const { calls, shell } = shellMock()
  const systemPreferences = {
    askForMediaAccess: async () => {
      throw new Error('unexpected prompt')
    }
  }

  const result = await requestMicrophoneAccess({ isMac: false, shell, systemPreferences })

  assert.deepEqual(result, { granted: true, settingsOpened: false, status: 'granted' })
  assert.deepEqual(calls, [])
})

test('macOS not-determined status prompts for microphone access first', async () => {
  const { calls, shell } = shellMock()
  const systemPreferences = {
    askForMediaAccess: async mediaType => {
      assert.equal(mediaType, 'microphone')

      return true
    },
    getMediaAccessStatus: () => 'not-determined'
  }

  const result = await requestMicrophoneAccess({ isMac: true, shell, systemPreferences })

  assert.deepEqual(result, { granted: true, settingsOpened: false, status: 'granted' })
  assert.deepEqual(calls, [])
})

test('macOS denied status opens microphone privacy settings', async () => {
  const { calls, shell } = shellMock()
  const systemPreferences = {
    askForMediaAccess: async () => true,
    getMediaAccessStatus: () => 'denied'
  }

  const result = await requestMicrophoneAccess({ isMac: true, shell, systemPreferences })

  assert.deepEqual(result, { granted: false, settingsOpened: true, status: 'denied' })
  assert.deepEqual(calls, [MICROPHONE_PRIVACY_SETTINGS_URL])
})

test('macOS prompt denial opens microphone privacy settings', async () => {
  const { calls, shell } = shellMock()
  let status = 'not-determined'
  const systemPreferences = {
    askForMediaAccess: async () => {
      status = 'denied'

      return false
    },
    getMediaAccessStatus: () => status
  }

  const result = await requestMicrophoneAccess({ isMac: true, shell, systemPreferences })

  assert.deepEqual(result, { granted: false, settingsOpened: true, status: 'denied' })
  assert.deepEqual(calls, [MICROPHONE_PRIVACY_SETTINGS_URL])
})

test('openMicrophonePrivacySettings is macOS-only', () => {
  const { calls, shell } = shellMock()

  assert.equal(openMicrophonePrivacySettings({ isMac: false, shell }), false)
  assert.deepEqual(calls, [])
})

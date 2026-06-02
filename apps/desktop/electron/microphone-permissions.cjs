const MICROPHONE_PRIVACY_SETTINGS_URL = 'x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone'

const KNOWN_MICROPHONE_STATUSES = new Set(['granted', 'denied', 'restricted', 'not-determined', 'unknown'])

function normalizeMicrophoneStatus(status) {
  const value = String(status || 'unknown')

  return KNOWN_MICROPHONE_STATUSES.has(value) ? value : 'unknown'
}

function readMicrophoneStatus(systemPreferences) {
  if (typeof systemPreferences?.getMediaAccessStatus !== 'function') {
    return 'unknown'
  }

  try {
    return normalizeMicrophoneStatus(systemPreferences.getMediaAccessStatus('microphone'))
  } catch {
    return 'unknown'
  }
}

function openMicrophonePrivacySettings({ isMac, rememberLog = () => {}, shell } = {}) {
  if (!isMac || typeof shell?.openExternal !== 'function') {
    return false
  }

  shell
    .openExternal(MICROPHONE_PRIVACY_SETTINGS_URL)
    .catch(error => rememberLog(`[permissions] failed to open microphone privacy settings: ${error.message}`))

  return true
}

async function requestMicrophoneAccess({ isMac, rememberLog = () => {}, shell, systemPreferences } = {}) {
  if (!isMac) {
    return { granted: true, settingsOpened: false, status: 'granted' }
  }

  let status = readMicrophoneStatus(systemPreferences)

  if (status === 'granted') {
    return { granted: true, settingsOpened: false, status }
  }

  if (
    (status === 'not-determined' || status === 'unknown') &&
    typeof systemPreferences?.askForMediaAccess === 'function'
  ) {
    let granted = false

    try {
      granted = await systemPreferences.askForMediaAccess('microphone')
    } catch (error) {
      rememberLog(`[permissions] microphone access prompt failed: ${error.message}`)
    }

    status = readMicrophoneStatus(systemPreferences)

    if (granted || status === 'granted') {
      return { granted: true, settingsOpened: false, status: status === 'granted' ? status : 'granted' }
    }
  }

  const settingsOpened = openMicrophonePrivacySettings({ isMac, rememberLog, shell })

  return { granted: false, settingsOpened, status }
}

module.exports = {
  MICROPHONE_PRIVACY_SETTINGS_URL,
  openMicrophonePrivacySettings,
  requestMicrophoneAccess
}

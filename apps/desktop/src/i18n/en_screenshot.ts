import type { ScreenshotTranslations } from './types_screenshot'

export const enScreenshot: ScreenshotTranslations = {
  leftCommand: 'Left ⌘',
  rightCommand: 'Right ⌘',
  enabledTitle: 'Screenshot shortcut',
  enabledDesc:
    'Press the left and right Command (⌘) keys together from any app to capture its frontmost window and attach it to your current Hermes draft. Never sends automatically. Off by default; applies only to this Mac. Window contents may be sensitive — review the attachment before sending.',
  statusTitle: 'Screenshot shortcut status',
  checking: 'Checking screenshot shortcut…',
  disabled: 'Screenshot shortcut is off.',
  starting: 'Starting the shortcut listener. It is not ready yet.',
  ready: 'Shortcut is ready. Screenshots attach to your current draft without sending.',
  inputPermission:
    'Input Monitoring permission lets Hermes detect the left and right Command (⌘) keys while another app is active. Allow Hermes in System Settings → Privacy & Security → Input Monitoring, then return here and retry.',
  screenPermission:
    'Screen Recording permission lets Hermes capture the frontmost app window when you use this shortcut. Allow Hermes in System Settings → Privacy & Security → Screen Recording, then return here and retry. Restart Hermes if macOS asks.',
  openSettings: 'Open System Settings',
  retry: 'Retry',
  unavailable: 'The screenshot shortcut is unavailable. Retry, or turn it off.',
  errorTitle: 'Screenshot shortcut error',
  loadFailed: 'Could not read the shortcut status. Retry to check its current setting.',
  saveFailed: 'Could not confirm the shortcut change. Retry to check its current setting.',
  permissionFailed: 'Could not open System Settings. Open Privacy & Security manually, then retry.',
  captureFailed: 'Could not capture the frontmost window. Nothing was attached or sent.',
  contextChanged: 'The current draft changed during capture. The screenshot was not attached or sent.'
}

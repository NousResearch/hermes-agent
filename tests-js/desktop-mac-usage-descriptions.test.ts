/**
 * Regression for #54551: macOS Info.plist privacy usage descriptions
 * declared by the Desktop electron-builder config
 * (`apps/desktop/electron-builder.config.cjs -> mac.extendInfo`) must pin every
 * `NS*UsageDescription` key the renderer relies on.
 *
 * Each entry is a key/value pair that lands in the packaged Hermes.app's
 * Info.plist via electron-builder's `extendInfo` merge. Missing or mis-stated
 * keys cause macOS to either silently deny the related API or surface a
 * mysteriously-worded system permission prompt at runtime (TCC's
 * `kTCCServiceMediaLibrary`, `kTCCServiceAppleEvents`, etc.).
 *
 * The Desktop renderer initializes Chromium's audio stack on user gesture
 * (completion chimes, TTS playback, voice mode). On macOS 26+, that init can
 * register the helper with the media subsystem and surface as a
 * "Hermes wants to access Music" prompt unless the Info.plist disclaims it
 * explicitly. This test pins every usage-description string the desktop
 * currently relies on so accidental drops break CI instead of breaking users.
 *
 * Why this test lives in tests-js/, not tests/*.py
 * -------------------------------------------------
 *
 * `AGENTS.md` requires assertions about JS-side packaging
 * artifacts to live in the JS/Vitest suite: the CI change classifier can
 * skip Python coverage on a JS-only PR (the classifier's `python` lane is
 * skipped when all paths match `_FRONTEND` or `_PY_SKIP`, both of which
 * cover `apps/desktop/package.json`). A regression would then go green on
 * the PR and red on `main` where the classifier fails open. See also
 * `tests-js/desktop-mac-entitlements.test.ts` which ports an earlier Python
 * entitlements regression for the same reason.
 *
 * Why this test exists
 * --------------------
 *
 * The project has a recurring class of bug: a macOS privacy-sensitive API is
 * called at runtime, but the Info.plist doesn't declare the corresponding
 * `NS*UsageDescription` key, so the system prompt is either silent (with a
 * generic "denied" error to the agent) or worded in a way that confuses the
 * user ("Hermes wants to access Music" when Hermes never touches the Music
 * library). The closed-PR family (#59486 / its duplicates #59833, #59915,
 * #59950, #60013 for Contacts; #39854 for Calendar; #64582 for Reminders)
 * established that the right fix shape is: add the key + pin it in a test.
 * This file is the canonical test for that pattern at the Desktop layer.
 *
 * When adding a new NS*UsageDescription key the runtime depends on, add a
 * matching row to EXPECTED_USAGE_DESCRIPTIONS below.
 */

import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')

const DESKTOP_CONFIG = path.join(
  REPO_ROOT,
  'apps',
  'desktop',
  'electron-builder.config.cjs'
)

const require = createRequire(import.meta.url)

interface UsageDescriptionRow {
  key: string
  reason: string
}

interface DesktopBuilderMacConfig {
  extendInfo?: object
}

interface DesktopBuilderConfig {
  mac?: DesktopBuilderMacConfig
}

const desktopBuilderConfig: DesktopBuilderConfig = require(DESKTOP_CONFIG)

function usageDescriptions(): Map<string, string> {
  const raw = desktopBuilderConfig.mac?.extendInfo
  assert.ok(
    typeof raw === 'object' && raw !== null && !Array.isArray(raw),
    'mac.extendInfo is missing or invalid in apps/desktop/electron-builder.config.cjs'
  )
  const result = new Map<string, string>()

  // electron-builder accepts arbitrary plist scalars in extendInfo, so only
  // narrow the usage-description subset this contract owns.
  for (const [key, value] of Object.entries(raw)) {
    if (!key.startsWith('NS') || !key.endsWith('UsageDescription')) {
      continue
    }

    assert.equal(
      typeof value,
      'string',
      `\`${key}\` in mac.extendInfo must be a string (got ${typeof value})`
    )
    result.set(key, value)
  }

  return result
}

// Each entry: Info.plist key and a plain-language reason. Catches silent
// drops of a key the runtime needs; the copy itself is free to change.
const EXPECTED_USAGE_DESCRIPTIONS: UsageDescriptionRow[] = [
  {
    key: 'NSMicrophoneUsageDescription',
    reason: 'Microphone capture is required for voice input mode.'
  },
  {
    key: 'NSAudioCaptureUsageDescription',
    reason: 'Audio capture backs the voice conversation pipeline.'
  },
  {
    key: 'NSCameraUsageDescription',
    reason: 'Camera access is requested by plugins/features the user enables.'
  },
  {
    key: 'NSAppleMusicUsageDescription',
    reason:
      "Disclaim MediaLibrary access so the system audio stack does not " +
      'surface a misleading Apple Music permission prompt ' +
      '(kTCCServiceMediaLibrary) when the renderer initializes audio for ' +
      'completion chimes, TTS, or voice.'
  },
  {
    key: 'NSCalendarsUsageDescription',
    reason: 'Calendar access backs meeting and scheduling support (#64571).'
  },
  {
    key: 'NSCalendarsFullAccessUsageDescription',
    reason: 'macOS 14+ full-access variant of the calendar declaration.'
  },
  {
    key: 'NSRemindersUsageDescription',
    reason: 'Reminders access backs personal-assistant scheduling (#64571).'
  },
  {
    key: 'NSRemindersFullAccessUsageDescription',
    reason: 'macOS 14+ full-access variant of the reminders declaration.'
  },
  {
    key: 'NSScreenCaptureUsageDescription',
    reason: 'macOS 15+ periodic screen-recording re-prompts show this copy.'
  },
  {
    key: 'NSLocalNetworkUsageDescription',
    reason:
      'macOS 15+ Local Network Privacy silently denies undeclared apps ' +
      '(#81563); declaration is required for the prompt to appear at all.'
  }
]

test.each(EXPECTED_USAGE_DESCRIPTIONS)(
  '`$key` is declared in mac.extendInfo',
  ({ key, reason }) => {
    const info = usageDescriptions()

    assert.ok(
      info.has(key),
      `Info.plist privacy usage description \`${key}\` is missing from ` +
        'apps/desktop/electron-builder.config.cjs mac.extendInfo. macOS will surface ' +
        'a misleading system prompt or silently deny the related API.\n' +
        `Reason: ${reason}`
    )
  }
)

test('every extendInfo value is free of leading/trailing whitespace and newlines', () => {
  const info = usageDescriptions()

  for (const [key, value] of info) {
    assert.equal(
      value,
      value.trim(),
      `\`${key}\` in mac.extendInfo has leading/trailing whitespace: ` +
        JSON.stringify(value)
    )
    // electron-builder writes strings as-is; newlines would render as
    // literal control chars in the macOS prompt.
    assert.ok(
      !value.includes('\n') && !value.includes('\r'),
      `\`${key}\` contains a newline; macOS will render it as a control ` +
        'character in the system permission prompt.'
    )
  }
})

/**
 * Regression for #54551: macOS Info.plist privacy usage descriptions
 * declared by the Desktop electron-builder config
 * (`apps/desktop/package.json -> build.mac.extendInfo`) must pin every
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
 * `AGENTS.md:1319-1329` requires assertions about `package.json` and JS-side
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
 * When adding a new NS*UsageDescription key to `build.mac.extendInfo`, add a
 * matching row to EXPECTED_USAGE_DESCRIPTIONS below. The drift-protection
 * assertion at the bottom of this file will fail otherwise.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const DESKTOP_PKG = path.join(REPO_ROOT, 'apps', 'desktop', 'package.json')

interface UsageDescriptionRow {
  key: string
  requiredSubstring: string
  reason: string
}

function desktopPkg(): Record<string, unknown> {
  assert.ok(fs.existsSync(DESKTOP_PKG), `missing ${DESKTOP_PKG}`)
  return JSON.parse(fs.readFileSync(DESKTOP_PKG, 'utf-8'))
}

function extendInfo(): Record<string, string> {
  const pkg = desktopPkg()
  const build = (pkg.build ?? {}) as Record<string, unknown>
  const mac = (build.mac ?? {}) as Record<string, unknown>
  assert.ok(
    typeof mac.extendInfo === 'object' &&
      mac.extendInfo !== null &&
      !Array.isArray(mac.extendInfo),
    'build.mac.extendInfo is missing or invalid in apps/desktop/package.json'
  )
  const extend = mac.extendInfo as Record<string, unknown>
  // Narrow to Record<string, string> with a runtime guard — the value type
  // for NS*UsageDescription is string, but electron-builder's `extendInfo`
  // accepts arbitrary plist scalars (bool, number, array, object) and we want
  // a clean assertion error here, not a downstream `value.trim is not a
  // function` crash in the whitespace test.
  for (const [key, value] of Object.entries(extend)) {
    assert.equal(
      typeof value,
      'string',
      `\`${key}\` in build.mac.extendInfo must be a string (got ${typeof value})`
    )
  }
  return extend as Record<string, string>
}

// Each entry: Info.plist key, required substring (case-insensitive), and a
// plain-language reason. The substring check lets future copy edits pass
// while still catching silent drops of the key itself.
const EXPECTED_USAGE_DESCRIPTIONS: UsageDescriptionRow[] = [
  {
    key: 'NSMicrophoneUsageDescription',
    requiredSubstring: 'microphone',
    reason: 'Microphone capture is required for voice input mode.'
  },
  {
    key: 'NSAudioCaptureUsageDescription',
    requiredSubstring: 'audio',
    reason: 'Audio capture backs the voice conversation pipeline.'
  },
  {
    key: 'NSAppleMusicUsageDescription',
    requiredSubstring: 'Music',
    reason:
      "Disclaim MediaLibrary access so the system audio stack does not " +
      'surface a misleading Apple Music permission prompt ' +
      '(kTCCServiceMediaLibrary) when the renderer initializes audio for ' +
      'completion chimes, TTS, or voice.'
  }
]

test.each(EXPECTED_USAGE_DESCRIPTIONS)(
  '`$key` is declared in build.mac.extendInfo',
  ({ key, requiredSubstring, reason }) => {
    const info = extendInfo()
    const value = info[key]

    assert.ok(
      value !== undefined,
      `Info.plist privacy usage description \`${key}\` is missing from ` +
        'apps/desktop/package.json build.mac.extendInfo. macOS will surface ' +
        'a misleading system prompt or silently deny the related API.\n' +
        `Reason: ${reason}`
    )

    assert.ok(
      value.toLowerCase().includes(requiredSubstring.toLowerCase()),
      `\`${key}\` exists but does not mention '${requiredSubstring}'. ` +
        `Current value: ${JSON.stringify(value)}. Reason: ${reason}`
    )
  }
)

test('every extendInfo value is free of leading/trailing whitespace and newlines', () => {
  const info = extendInfo()
  for (const [key, value] of Object.entries(info)) {
    assert.equal(
      value,
      value.trim(),
      `\`${key}\` in build.mac.extendInfo has leading/trailing whitespace: ` +
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

test('every NS*UsageDescription in extendInfo is pinned in this test', () => {
  const info = extendInfo()
  const declaredKeys = new Set(EXPECTED_USAGE_DESCRIPTIONS.map((row) => row.key))
  // Non-privacy keys (CFBundleDisplayName etc.) are exempt — this test
  // only governs NS*UsageDescription entries.
  const privacyKeysInPlist = new Set(
    Object.keys(info).filter(
      (k) => k.startsWith('NS') && k.endsWith('UsageDescription')
    )
  )
  const missing = [...privacyKeysInPlist].filter((k) => !declaredKeys.has(k))
  assert.deepEqual(
    missing,
    [],
    `extendInfo declares privacy usage keys ${JSON.stringify(missing.sort())} ` +
      'that this test does not pin. Add them to EXPECTED_USAGE_DESCRIPTIONS ' +
      'with a reason, or remove them from the build config.'
  )
})
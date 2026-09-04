// gen-appinstaller — the .appinstaller document for an out-of-store channel.
// The identity inside must match the package manifest (same derivation), and
// the bundle/package URLs must resolve under the feed host.
import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { OUT_OF_STORE_PUBLISHER, buildAppInstaller } from '../../../scripts/msix-shared.mjs'

describe('buildAppInstaller', () => {
  const base = {
    baseUrl: 'https://updates.example.com',
    variantChannelPath: 'win32/stable',
    identityName: 'NousResearch.HermesBundled',
    version: '0.3.0.0',
    bundleFilename: 'HermesBundled-0.3.0.0-win.msixbundle'
  }

  test('pins the same publisher as the out-of-store manifest (ATS cert subject)', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /Publisher="CN=Nous Research Inc\., O=Nous Research Inc\., L=Austin, S=Texas, C=US"/)
    assert.equal(OUT_OF_STORE_PUBLISHER, 'CN=Nous Research Inc., O=Nous Research Inc., L=Austin, S=Texas, C=US')
  })

  test('MainPackage URI and the canonical Uri point at the bundle + appinstaller under the feed dir', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /Uri="https:\/\/updates\.example\.com\/win32\/stable\/HermesBundled-0\.3\.0\.0-win\.msixbundle"/)
    // The AppInstaller's own Uri is the bundle URL with .appinstaller swapped in.
    assert.match(xml, new RegExp(`^<AppInstaller\\n  Uri="[^"]+/win32/stable/${base.bundleFilename.replace(/\.msixbundle$/, '.appinstaller')}"`, 'm'))
  })

  test('MainPackage Name equals the package identity; version matches everywhere', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /Name="NousResearch\.HermesBundled"/)
    const versionCount = (xml.match(/Version="0\.3\.0\.0"/g) || []).length
    // AppInstaller Version + MainPackage Version = 2 occurrences.
    assert.equal(versionCount, 2)
  })

  test('UpdateSettings keeps the OS prompt off (the in-app checker owns the prompt)', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /<OnLaunch HoursBetweenUpdateChecks="12" ShowPrompt="false" \/>/)
  })

  test('a variant channel path with a trailing slash still resolves under the host', () => {
    const xml = buildAppInstaller({ ...base, variantChannelPath: 'win32/canary/' })
    assert.match(xml, /https:\/\/updates\.example\.com\/win32\/canary\//)
  })

  test('reserved XML characters in identity values are escaped', () => {
    const xml = buildAppInstaller({ ...base, identityName: 'A&B<App>' })
    assert.match(xml, /Name="A&amp;B&lt;App&gt;"/)
  })
})

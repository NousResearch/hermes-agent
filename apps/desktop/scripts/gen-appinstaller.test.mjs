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

  test('MainBundle points at the package bytes and self URI stays on the published channel descriptor', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /<MainBundle\s/)
    assert.doesNotMatch(xml, /<MainPackage\s/)
    assert.match(xml, /Uri="https:\/\/updates\.example\.com\/win32\/stable\/HermesBundled-0\.3\.0\.0-win\.msixbundle"/)
    const descriptor = /<AppInstaller\s+Uri="([^"]+)"/.exec(xml)[1]
    assert.equal(descriptor, `${base.baseUrl}/${base.variantChannelPath}/stable.appinstaller`)
    const next = buildAppInstaller({ ...base, version: '0.3.1.0', bundleFilename: 'next.msixbundle' })
    assert.equal(/<AppInstaller\s+Uri="([^"]+)"/.exec(next)[1], descriptor)
  })

  test('MainBundle Name equals the package identity; version matches everywhere', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /Name="NousResearch\.HermesBundled"/)
    const versionCount = (xml.match(/Version="0\.3\.0\.0"/g) || []).length
    // AppInstaller Version + MainBundle Version = 2 occurrences.
    assert.equal(versionCount, 2)
  })

  test('UpdateSettings keeps the OS prompt off (the in-app checker owns the prompt)', () => {
    const xml = buildAppInstaller(base)
    assert.match(xml, /<OnLaunch HoursBetweenUpdateChecks="12" \/>/)
    assert.doesNotMatch(xml, /ShowPrompt=/)
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

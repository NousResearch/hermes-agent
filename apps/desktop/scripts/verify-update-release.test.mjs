import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, test } from 'vitest'

import { readPackagedPublisherNames, verifyUpdateRelease } from './verify-update-release.mjs'

const tempDirs = []

function makeFixture() {
  const releaseDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-release-'))
  tempDirs.push(releaseDir)

  const version = '1.2.3'
  const installerName = `Hermes-Setup-${version}.exe`
  const installer = Buffer.from('signed-installer-fixture')
  const sha512 = crypto.createHash('sha512').update(installer).digest('base64')

  fs.writeFileSync(path.join(releaseDir, installerName), installer)
  fs.writeFileSync(path.join(releaseDir, `${installerName}.blockmap`), '{}')
  fs.writeFileSync(
    path.join(releaseDir, 'latest.yml'),
    [
      `version: ${version}`,
      'files:',
      `  - url: ${installerName}`,
      `    sha512: ${sha512}`,
      `    size: ${installer.length}`,
      `path: ${installerName}`,
      `sha512: ${sha512}`,
      "releaseDate: '2026-07-28T00:00:00.000Z'",
      ''
    ].join('\n')
  )

  return { installerName, releaseDir, version }
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('update release verification', () => {
  test('accepts a synchronized installer, blockmap, and latest.yml', async () => {
    const fixture = makeFixture()
    const result = await verifyUpdateRelease({
      expectedVersion: fixture.version,
      releaseDir: fixture.releaseDir
    })

    assert.equal(result.version, fixture.version)
    assert.equal(result.installer, fixture.installerName)
    assert.equal(result.signature, 'not-required')
  })

  test('rejects metadata version drift', async () => {
    const fixture = makeFixture()

    await assert.rejects(
      verifyUpdateRelease({ expectedVersion: '1.2.4', releaseDir: fixture.releaseDir }),
      /latest\.yml version 1\.2\.3 does not match expected 1\.2\.4/
    )
  })

  test('rejects installer hash mismatches', async () => {
    const fixture = makeFixture()
    fs.appendFileSync(path.join(fixture.releaseDir, fixture.installerName), 'tampered')

    await assert.rejects(
      verifyUpdateRelease({ expectedVersion: fixture.version, releaseDir: fixture.releaseDir }),
      /installer size mismatch|installer SHA-512 mismatch/
    )
  })

  test('requires the matching differential blockmap', async () => {
    const fixture = makeFixture()
    fs.rmSync(path.join(fixture.releaseDir, `${fixture.installerName}.blockmap`))

    await assert.rejects(
      verifyUpdateRelease({ expectedVersion: fixture.version, releaseDir: fixture.releaseDir }),
      /missing blockmap/
    )
  })

  test('requires the requested packaged executable', async () => {
    const fixture = makeFixture()
    const packagedExecutable = path.join(fixture.releaseDir, 'win-unpacked', 'Hermes.exe')

    await assert.rejects(
      verifyUpdateRelease({
        expectedVersion: fixture.version,
        packagedExecutable,
        releaseDir: fixture.releaseDir
      }),
      /missing packaged executable/
    )
  })

  test('requires both packaged trust artifacts in production signature mode', async () => {
    const fixture = makeFixture()

    await assert.rejects(
      verifyUpdateRelease({
        expectedVersion: fixture.version,
        releaseDir: fixture.releaseDir,
        requireSignature: true
      }),
      /--require-signature requires a packaged executable/
    )

    const packagedExecutable = path.join(fixture.releaseDir, 'Hermes.exe')
    fs.writeFileSync(packagedExecutable, 'packaged-executable-fixture')

    await assert.rejects(
      verifyUpdateRelease({
        expectedVersion: fixture.version,
        packagedExecutable,
        releaseDir: fixture.releaseDir,
        requireSignature: true
      }),
      /--require-signature requires a packaged updater configuration/
    )
  })

  test('reads runtime publisher identities from packaged app-update.yml', () => {
    const fixture = makeFixture()
    const updateConfig = path.join(fixture.releaseDir, 'app-update.yml')
    fs.writeFileSync(
      updateConfig,
      ['provider: github', 'publisherName:', "  - 'CN=Nous Research, O=Nous Research'", '  - Nous Research', ''].join(
        '\n'
      )
    )

    assert.deepEqual(readPackagedPublisherNames(updateConfig), ['CN=Nous Research, O=Nous Research', 'Nous Research'])
  })

  test('rejects packaged updater configuration that disables runtime signature verification', () => {
    const fixture = makeFixture()
    const updateConfig = path.join(fixture.releaseDir, 'app-update.yml')
    fs.writeFileSync(updateConfig, ['provider: github', 'owner: NousResearch', 'repo: hermes-agent', ''].join('\n'))

    assert.throws(() => readPackagedPublisherNames(updateConfig), /does not contain a non-empty publisherName/)
  })

  test('rejects metadata paths that escape the release directory', async () => {
    const fixture = makeFixture()
    const latestPath = path.join(fixture.releaseDir, 'latest.yml')
    fs.writeFileSync(latestPath, fs.readFileSync(latestPath, 'utf8').replaceAll(fixture.installerName, '../evil.exe'))

    await assert.rejects(
      verifyUpdateRelease({ expectedVersion: fixture.version, releaseDir: fixture.releaseDir }),
      /must be a plain filename/
    )
  })
})

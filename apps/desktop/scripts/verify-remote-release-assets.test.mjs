import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, test } from 'vitest'

import { verifyRemoteReleaseAssets } from './verify-remote-release-assets.mjs'

const tempDirs = []

function sha256(content) {
  return crypto.createHash('sha256').update(content).digest('hex')
}

function fixture() {
  const stagingDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-remote-assets-'))
  tempDirs.push(stagingDir)
  const installer = Buffer.from('signed-installer')
  const latest = Buffer.from('version: 1.2.3\n')
  fs.writeFileSync(path.join(stagingDir, 'Hermes-Setup-1.2.3.exe'), installer)
  fs.writeFileSync(path.join(stagingDir, 'latest.yml'), latest)

  return {
    installer,
    latest,
    remoteRelease: {
      assets: [
        {
          digest: `sha256:${sha256(installer)}`,
          name: 'Hermes-Setup-1.2.3.exe',
          size: installer.length
        },
        { digest: `sha256:${sha256(latest)}`, name: 'latest.yml', size: latest.length }
      ]
    },
    stagingDir
  }
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('remote GitHub release asset verification', () => {
  test('accepts an exact name, size, and GitHub SHA-256 digest match', () => {
    const data = fixture()

    assert.deepEqual(verifyRemoteReleaseAssets(data), {
      assetCount: 2,
      names: ['Hermes-Setup-1.2.3.exe', 'latest.yml']
    })
  })

  test('rejects a same-length remote replacement with a different digest', () => {
    const data = fixture()
    data.remoteRelease.assets[0].digest = `sha256:${sha256(Buffer.from('evil-installer!!'))}`

    assert.throws(() => verifyRemoteReleaseAssets(data), /SHA-256 digest mismatch for Hermes-Setup-1\.2\.3\.exe/)
  })

  test('fails closed when GitHub does not provide a SHA-256 digest', () => {
    const data = fixture()
    data.remoteRelease.assets[0].digest = null

    assert.throws(() => verifyRemoteReleaseAssets(data), /missing GitHub SHA-256 digest/)
  })

  test('can verify the immutable set before latest.yml is uploaded', () => {
    const data = fixture()
    data.remoteRelease.assets = data.remoteRelease.assets.filter(asset => asset.name !== 'latest.yml')

    assert.deepEqual(verifyRemoteReleaseAssets({ ...data, excludeNames: ['latest.yml'] }), {
      assetCount: 1,
      names: ['Hermes-Setup-1.2.3.exe']
    })
  })
})

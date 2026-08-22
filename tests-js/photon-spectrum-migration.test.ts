import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const SIDECAR = path.join(REPO_ROOT, 'plugins', 'platforms', 'photon', 'sidecar')
const SIDECAR_PACKAGE = path.join(SIDECAR, 'package.json')
const SIDECAR_LOCK = path.join(SIDECAR, 'package-lock.json')
const PATCH_NAME = 'patch-spectrum-mixed-attachments.mjs'

type PackageJson = {
  dependencies?: Record<string, string>
  engines?: Record<string, string>
  scripts?: Record<string, string>
}

type Lockfile = {
  packages?: Record<string, { version?: string }>
}

function packageJson(): PackageJson {
  return JSON.parse(fs.readFileSync(SIDECAR_PACKAGE, 'utf-8'))
}

function lockfile(): Lockfile {
  return JSON.parse(fs.readFileSync(SIDECAR_LOCK, 'utf-8'))
}

function major(version: string): number {
  const match = version.match(/\d+/)
  assert.ok(match, `Expected a version containing a major number, received ${version}`)

  return Number(match[0])
}

test('Photon sidecar locks its declared Spectrum generation', () => {
  const dependencies = packageJson().dependencies ?? {}
  const spectrum = dependencies['spectrum-ts']
  const packages = lockfile().packages ?? {}

  assert.ok(spectrum, 'Photon sidecar must declare spectrum-ts directly.')
  assert.ok(
    major(spectrum) >= 12,
    'Photon media delivery requires Spectrum 12 or later; Spectrum 8 rejects valid media sends.'
  )
  assert.equal(
    packages['node_modules/spectrum-ts']?.version,
    spectrum,
    'The sidecar lockfile must resolve the exact Spectrum version declared by package.json.'
  )
  assert.equal(
    packages['node_modules/@spectrum-ts/imessage']?.version,
    spectrum,
    'The iMessage provider must resolve with the same Spectrum release as the sidecar umbrella package.'
  )
  assert.ok(
    'ffmpeg-static' in dependencies && 'node_modules/ffmpeg-static' in packages,
    'Photon native voice delivery converts MP3 TTS to AAC/M4A and requires ffmpeg-static in both manifest and lockfile.'
  )
})

test('Photon sidecar uses native mixed-part handling without the Spectrum 8 patch', () => {
  const sidecarPackage = packageJson()

  const runtimeSources = [
    path.join(REPO_ROOT, 'plugins', 'platforms', 'photon', 'adapter.py'),
    path.join(SIDECAR, 'index.mjs'),
    path.join(REPO_ROOT, 'plugins', 'platforms', 'photon', 'sidecar_paths.py'),
    path.join(REPO_ROOT, 'Dockerfile'),
  ]

  assert.ok(!('postinstall' in (sidecarPackage.scripts ?? {})))
  assert.ok(!fs.existsSync(path.join(SIDECAR, PATCH_NAME)))

  for (const source of runtimeSources) {
    assert.ok(
      !fs.readFileSync(source, 'utf-8').includes(PATCH_NAME),
      `${path.relative(REPO_ROOT, source)} must not reference the removed Spectrum 8 patch.`
    )
  }
})

test('Photon sidecar Node floor is met by the container runtime', () => {
  const nodeRange = packageJson().engines?.node
  const dockerfile = fs.readFileSync(path.join(REPO_ROOT, 'Dockerfile'), 'utf-8')
  const nodeImage = dockerfile.match(/FROM node:(\d+)/)

  assert.ok(nodeRange, 'Photon sidecar must declare its Node engine floor.')
  assert.ok(nodeImage, 'Dockerfile must declare its Node runtime source image.')
  assert.ok(
    major(nodeImage[1]) >= major(nodeRange),
    'The Docker Node image must meet the Photon sidecar engine floor.'
  )
})

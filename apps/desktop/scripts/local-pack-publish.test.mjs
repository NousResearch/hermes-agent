/**
 * Regression #87758: a local desktop pack must never enter electron-builder's
 * publish path, and publish resolution must succeed when something else does
 * enter it.
 *
 * These call the real app-builder-lib resolver rather than asserting on the
 * text of the pack script, so they fail if electron-builder changes the
 * behavior we depend on — not when someone reformats package.json.
 */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, beforeEach, describe, test } from 'vitest'

const require = createRequire(import.meta.url)
const desktopDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const desktopPkg = require(path.join(desktopDir, 'package.json'))

const { getPublishConfigs } = require('app-builder-lib/out/publish/PublishManager.js')
const { getRepositoryInfo } = require('app-builder-lib/out/util/repositoryInfo.js')

/**
 * The slice of PlatformPackager that getPublishConfigs actually reads. The
 * repositoryInfo getter mirrors what electron-builder does for real: resolve
 * from THIS package's metadata with projectDir = apps/desktop.
 */
function fakePackager(metadata) {
  const info = {
    get repositoryInfo() {
      return getRepositoryInfo(desktopDir, metadata, null)
    },
    appInfo: { version: '0.0.0', channel: null, updaterCacheDirName: 'hermes' },
    config: {},
    options: {}
  }
  return {
    platformSpecificBuildOptions: {},
    config: {},
    info,
    platform: { name: 'linux' },
    appInfo: info.appInfo,
    expandMacro: value => value
  }
}

const TOKEN_VARS = ['GH_TOKEN', 'GITHUB_TOKEN', 'GITLAB_TOKEN', 'KEYGEN_TOKEN', 'BITBUCKET_TOKEN']
let savedEnv

beforeEach(() => {
  savedEnv = {}
  for (const key of TOKEN_VARS) {
    savedEnv[key] = process.env[key]
    delete process.env[key]
  }
})

afterEach(() => {
  for (const key of TOKEN_VARS) {
    if (savedEnv[key] === undefined) delete process.env[key]
    else process.env[key] = savedEnv[key]
  }
})

describe('local desktop pack stays out of the publish path', () => {
  test('the pack script pins an explicit publish policy', () => {
    // electron-builder 26 infers `onTagOrDraft` from CI when --publish is
    // absent, and `hermes desktop` runs the pack with CI=1 (_npm_lifecycle_env).
    // v27 drops the implicit behavior, so being explicit is also forward-safe.
    const pack = desktopPkg.scripts.pack
    assert.match(pack, /--dir\b/)
    assert.match(pack, /--publish\s+never\b/)
  })

  test('publish resolution succeeds with a GitHub token present', async () => {
    // The #87758 failure mode: CI=1 makes isPublish true, a GITHUB_TOKEN in the
    // environment auto-selects the github provider, and the provider needs a
    // repository it cannot find — apps/desktop has no .git/config of its own
    // and app-builder-lib does not walk up to the workspace root.
    process.env.GITHUB_TOKEN = 'x'

    const configs = await getPublishConfigs(
      fakePackager(desktopPkg),
      null,
      null,
      /* errorIfCannot */ true
    )

    assert.ok(Array.isArray(configs) && configs.length > 0)
    assert.equal(configs[0].provider, 'github')
    assert.equal(configs[0].owner, 'NousResearch')
    assert.equal(configs[0].repo, 'hermes-agent')
  })

  test('a package without the repository field is what breaks resolution', async () => {
    // Guards the fix itself: proves the assertion above passes because of the
    // repository field, not because the throw is unreachable.
    process.env.GITHUB_TOKEN = 'x'
    const { repository, ...withoutRepository } = desktopPkg
    assert.ok(repository, 'apps/desktop/package.json must declare a repository')

    await assert.rejects(
      () => getPublishConfigs(fakePackager(withoutRepository), null, null, true),
      /Cannot detect repository by \.git\/config/
    )
  })

  test('resolution is quiet when no publish token is configured', async () => {
    const configs = await getPublishConfigs(fakePackager(desktopPkg), null, null, true)
    assert.deepEqual(configs, [])
  })
})

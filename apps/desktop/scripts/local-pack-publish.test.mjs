import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { LinuxPackager, Packager } from 'app-builder-lib'
import { afterEach, test, vi } from 'vitest'

const require = createRequire(import.meta.url)
const lib = path.dirname(path.dirname(require.resolve('app-builder-lib')))
const { getPublishConfigs } = require(path.join(lib, 'dist/publish/PublishManager.js'))
afterEach(() => vi.unstubAllEnvs())

test.each([['repository', true, true], ['missing repository', false, true], ['no token', true, false]])(
  'real supplier publish resolution: %s', async (_name, repository, token) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'publish-resolution-'))
    for (const key of ['GH_TOKEN', 'GITHUB_TOKEN', 'GITLAB_TOKEN', 'KEYGEN_TOKEN', 'BITBUCKET_TOKEN']) vi.stubEnv(key, '')
    vi.stubEnv('GITHUB_TOKEN', token ? 'fixture-not-a-real-token' : '')
    for (const key of ['TRAVIS_REPO_SLUG', 'APPVEYOR_REPO_NAME', 'CIRCLE_PROJECT_USERNAME', 'CIRCLE_PROJECT_REPONAME']) vi.stubEnv(key, undefined)
    // No .git parent and no production metadata resolver replaced. Only resolution, never publish.
    try {
      /** @type {import('app-builder-lib').Metadata} */
      const metadata = { name: 'fixture', version: '1.0.0', description: 'fixture', author: 'Fixture' }
      if (repository) metadata.repository = 'https://github.com/NousResearch/hermes-agent'
      fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify(metadata))
      const info = new Packager({ projectDir: root, config: {} })
      await info.validateConfig()
      const packager = new LinuxPackager(info)
      if (!token) {
        assert.deepEqual(await getPublishConfigs(packager, null, null, true), [])
      } else if (!repository) {
        await assert.rejects(getPublishConfigs(packager, null, null, true), /Cannot detect repository/)
      } else {
        const repositoryInfo = await packager.repositoryInfo
        assert.equal(repositoryInfo.user, 'NousResearch')
        assert.equal(repositoryInfo.project, 'hermes-agent')
        assert.deepEqual(await getPublishConfigs(packager, null, null, true), [{ owner: 'NousResearch', repo: 'hermes-agent', provider: 'github' }])
      }
    } finally { fs.rmSync(root, { recursive: true, force: true }) }
  }
)

test('the pack script pins an explicit publish policy', () => {
  const pack = require('../package.json').scripts.pack
  assert.match(pack, /--dir\b/)
  assert.match(pack, /--publish\s+never\b/)
})

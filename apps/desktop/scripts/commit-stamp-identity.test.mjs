import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

const root = path.resolve(import.meta.dirname, '../../..')

test('desktop stamp uses the admitted checkout rather than the dispatch SHA', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'stamp-identity-'))
  const repo = path.join(temp, 'repo')
  const scripts = path.join(repo, 'apps/desktop/scripts')
  const git = (...args) => execFileSync('git', args, { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim()
  try {
    fs.mkdirSync(scripts, { recursive: true })
    for (const name of ['write-build-stamp.mjs', 'utils.mjs']) fs.copyFileSync(path.join(root, 'apps/desktop/scripts', name), path.join(scripts, name))
    fs.copyFileSync(path.join(root, 'apps/desktop/product-identity.cjs'), path.join(repo, 'apps/desktop/product-identity.cjs'))
    git('init', '-q', '-b', 'main')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', '.')
    git('commit', '-qm', 'main')
    const main = git('rev-parse', 'HEAD')
    git('checkout', '-qb', 'feature')
    fs.writeFileSync(path.join(repo, 'feature'), 'feature\n')
    git('add', 'feature')
    git('commit', '-qm', 'feature')
    const feature = git('rev-parse', 'HEAD')
    const runtime = { repoDir: 'app', toolsDir: 'tools', storePython: 'tools/python/bin/python3', sitePackages: 'venv/site-packages', commands: { hermes: 'bin/hermes' } }
    const build = path.join(repo, 'apps/desktop/build')
    fs.mkdirSync(path.join(build, 'agent-payload/app'), { recursive: true })
    fs.mkdirSync(path.join(build, 'agent-payload/bin'), { recursive: true })
    fs.writeFileSync(path.join(build, 'agent-payload/bin/hermes'), 'launcher fixture')
    fs.writeFileSync(path.join(build, 'agent-payload/manifest.json'), JSON.stringify({ target: 'darwin-arm64', launchers: ['hermes'], runtime }))
    const out = path.join(build, 'install-stamp.json')
    const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('GITHUB_') && !key.startsWith('HERMES_PAYLOAD_') && !key.startsWith('HERMES_BUILD_')))
    Object.assign(env, { GITHUB_SHA: main, GITHUB_REF_NAME: 'main', HERMES_BUILD_COMMIT: feature, HERMES_DESKTOP_VARIANT: 'bundled' })
    const run = override => spawnSync(process.execPath, [path.join(scripts, 'write-build-stamp.mjs')], { cwd: temp, env: { ...env, ...override }, encoding: 'utf8', timeout: 30000 })
    const result = run({})
    assert.equal(result.status, 0, result.stderr)
    const stamp = JSON.parse(fs.readFileSync(out, 'utf8'))
    assert.equal(stamp.commit, feature)
    assert.equal(stamp.branch, null)
    assert.equal(stamp.source, 'commit-build')
    assert.equal(stamp.tag, null)
    assert.equal(stamp.updateMechanism, 'external')
    assert.deepEqual(stamp.runtime, { ...runtime, commands: { hermes: `bin/hermes-${feature.slice(0, 7)}` } })
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(build, 'agent-payload/app/install-stamp.json'), 'utf8')), stamp)
    const before = fs.readFileSync(out)
    git('checkout', '-q', 'main')
    assert.notEqual(run({}).status, 0)
    assert.deepEqual(fs.readFileSync(out), before)
    git('checkout', '-q', 'feature')
    for (const override of [{ HERMES_BUILD_COMMIT: feature.slice(0, 8) }, { HERMES_BUILD_COMMIT: ` ${feature}` }, { HERMES_PAYLOAD_TAG: 'v1.2.3' }]) {
      assert.notEqual(run(override).status, 0)
      assert.deepEqual(fs.readFileSync(out), before)
    }
    assert.equal(run({ HERMES_BUILD_COMMIT: '', HERMES_PAYLOAD_TAG: 'v1.2.3', GITHUB_SHA: feature }).status, 0)
    assert.equal(JSON.parse(fs.readFileSync(out, 'utf8')).tag, 'v1.2.3')
    assert.deepEqual(JSON.parse(fs.readFileSync(out, 'utf8')).runtime, runtime)
  } finally {
    fs.rmSync(temp, { recursive: true, force: true })
  }
})

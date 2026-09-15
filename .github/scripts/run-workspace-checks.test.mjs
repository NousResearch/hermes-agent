import assert from 'node:assert/strict'
import { execFileSync, spawn } from 'node:child_process'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import test from 'node:test'

const runner = fileURLToPath(new URL('./run-workspace-checks.mjs', import.meta.url))

async function runFixture(t, source, { check = true, deferredOutput = false } = {}) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'hermes-workspace-checks-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const workspace = path.join(directory, 'fixture')
  await mkdir(workspace)
  await writeFile(path.join(directory, 'package.json'), JSON.stringify({ private: true, workspaces: ['fixture'] }))
  await writeFile(
    path.join(workspace, 'package.json'),
    JSON.stringify({ name: 'fixture', version: '1.0.0', scripts: check ? { check: 'node fixture.mjs' } : {} })
  )
  await writeFile(path.join(workspace, 'fixture.mjs'), source)
  // npm query inspects the installed workspace link. There are no third-party
  // dependencies; offline installation creates only this temporary fixture.
  execFileSync(
    process.platform === 'win32' ? 'npm.cmd' : 'npm',
    ['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund'],
    {
      cwd: directory,
      shell: process.platform === 'win32',
      stdio: 'pipe',
      windowsHide: true
    }
  )
  const preload = path.join(directory, 'deferred-stdout.mjs')
  // Unix CI pipes are asynchronous, while Windows pipes may drain
  // synchronously. Model the pending-write boundary explicitly on either OS.
  await writeFile(
    preload,
    `
    const write = process.stdout.write.bind(process.stdout)
    process.stdout.write = (...args) => {
      setTimeout(() => write(...args), 50)
      return false
    }
  `
  )
  const args = [...(deferredOutput ? ['--import', pathToFileURL(preload).href] : []), runner, '--concurrency', '1']
  const child = spawn(process.execPath, args, {
    cwd: directory,
    env: { ...process.env, GITHUB_ACTIONS: 'true' },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true
  })
  t.after(() => {
    if (child.exitCode === null) child.kill()
  })
  let stdout = ''
  let stderr = ''
  child.stdout.setEncoding('utf8')
  child.stderr.setEncoding('utf8')
  child.stdout.on('data', chunk => {
    stdout += chunk
    // Model a CI log consumer applying backpressure to a verbose failure.
    child.stdout.pause()
    setTimeout(() => child.stdout.resume(), 2)
  })
  child.stderr.on('data', chunk => {
    stderr += chunk
  })
  const code = await new Promise((resolve, reject) => {
    child.once('error', reject)
    child.once('close', resolve)
  })
  return { code, stdout, stderr }
}

test('preserves the end of a verbose failing check and its nonzero result', { timeout: 30_000 }, async t => {
  const marker = 'END_OF_FAILING_WORKSPACE_OUTPUT'
  const result = await runFixture(
    t,
    `process.stdout.write('x'.repeat(2 * 1024 * 1024) + '\\n${marker}\\n'); process.exitCode = 7`,
    { deferredOutput: true }
  )
  assert.equal(result.code, 1)
  assert.ok(result.stdout.includes(marker), 'the final failure diagnostic must survive buffered output')
  assert.ok(result.stdout.includes('=== summary ==='))
  assert.match(result.stderr, /1 of 1 checks failed/)
})

test('retains a successful workspace result', { timeout: 30_000 }, async t => {
  const result = await runFixture(t, "console.log('CHECK_SUCCEEDED')")
  assert.equal(result.code, 0, (result.stdout + result.stderr).slice(-2400))
  assert.match(result.stdout, /CHECK_SUCCEEDED/)
  assert.match(result.stdout, /all 1 checks passed/)
})

test('fails closed when no workspace declares a check', { timeout: 30_000 }, async t => {
  const result = await runFixture(t, '', { check: false })
  assert.equal(result.code, 1)
  assert.match(result.stderr, /refusing to report green having run nothing/)
})

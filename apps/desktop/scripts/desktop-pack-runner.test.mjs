import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { recoverDesktopBuilder, runDesktopBuilder } from './desktop-pack-runner.mjs'
import { PACK_SESSION_ENV, PACK_JOURNAL_ENV, writeRollbackSession } from './desktop-pack-transaction.mjs'
import { peFixture } from './pe-test-fixture.mjs'

const hook = new URL('./before-pack.mjs', import.meta.url).href

for (const scenario of ['success', 'failure', 'invalid', 'missing-first-install', 'interrupted-retry']) {
  test(`real builder process settles custom output: ${scenario}`, () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pack-process-'))
    const output = path.join(root, 'custom output-é', 'win-arm64-unpacked')
    const original = peFixture('original')
    const replacement = peFixture('replacement')
    const backup = `${output}.bak`
    try {
      if (scenario !== 'missing-first-install') {
        fs.mkdirSync(output, { recursive: true })
        fs.writeFileSync(path.join(output, 'Hermes.exe'), original)
      }
      if (scenario === 'interrupted-retry') {
        fs.renameSync(output, backup)
        fs.mkdirSync(output)
        fs.writeFileSync(path.join(output, 'Hermes.exe'), 'MZ-truncated')
      }
      const script = path.join(root, 'builder.mjs')
      fs.writeFileSync(script, `
        import fs from 'node:fs';
        import path from 'node:path';
        import beforePack from ${JSON.stringify(hook)};
        const output = process.env.FIXTURE_OUTPUT;
        await beforePack({ appOutDir: output, electronPlatformName: 'win32' });
        fs.mkdirSync(output, { recursive: true });
        const scenario = process.env.FIXTURE_SCENARIO;
        if (scenario !== 'missing-first-install') fs.writeFileSync(path.join(output, 'Hermes.exe'),
          scenario === 'success' ? Buffer.from(process.env.FIXTURE_PE, 'base64') : 'MZ-truncated');
        process.exit(scenario === 'failure' || scenario === 'interrupted-retry' ? 7 : 0);
      `)
      const result = runDesktopBuilder(process.execPath, [script], { env: {
        ...process.env, FIXTURE_OUTPUT: output, FIXTURE_SCENARIO: scenario,
        FIXTURE_PE: replacement.toString('base64')
      } })
      assert.equal(result.status, scenario === 'success' ? 0 : scenario === 'failure' || scenario === 'interrupted-retry' ? 7 : 1)
      if (scenario === 'success') {
        assert.deepEqual(fs.readFileSync(path.join(output, 'Hermes.exe')), replacement)
        assert.deepEqual(fs.readFileSync(path.join(backup, 'Hermes.exe')), original)
      } else if (scenario !== 'missing-first-install') {
        assert.deepEqual(fs.readFileSync(path.join(output, 'Hermes.exe')), original)
      }
      if (scenario === 'missing-first-install') assert.equal(result.settlement.ok, false)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }, 30000)
}

test('caller-owned identity survives the wrapper and permits recovery after interruption', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-pack-caller-'))
  const output = path.join(root, 'win-unpacked')
  const journal = path.join(root, 'targets.jsonl')
  const session = 'caller-operation'
  try {
    fs.mkdirSync(`${output}.bak`)
    fs.writeFileSync(path.join(`${output}.bak`, 'Hermes.exe'), peFixture('original'))
    writeRollbackSession(`${output}.bak`, session)
    fs.writeFileSync(journal, JSON.stringify({ appOutDir: output, productExeName: 'Hermes.exe' }) + '\n')
    const result = runDesktopBuilder(process.execPath, [], {
      env: { ...process.env, [PACK_SESSION_ENV]: session, [PACK_JOURNAL_ENV]: journal },
      spawn(_exe, _args, options) {
        assert.equal(options.env[PACK_SESSION_ENV], session)
        assert.equal(options.env[PACK_JOURNAL_ENV], journal)
        assert.notEqual(fs.readFileSync(journal, 'utf8'), '')
        return { status: 7 }
      }
    })
    assert.equal(result.status, 7)
    assert.ok(fs.existsSync(journal))
    assert.deepEqual(fs.readFileSync(path.join(output, 'Hermes.exe')), peFixture('original'))
    // A later caller recovery is idempotent and cannot consume a foreign backup.
    assert.equal(recoverDesktopBuilder(journal, session).ok, true)
    fs.renameSync(output, `${output}.bak`)
    writeRollbackSession(`${output}.bak`, 'different-operation')
    assert.equal(recoverDesktopBuilder(journal, session).restored.length, 0)
    assert.ok(fs.existsSync(`${output}.bak`))
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})

test('a partial caller identity fails before launching a builder', () => {
  const result = runDesktopBuilder(process.execPath, [], {
    env: { [PACK_SESSION_ENV]: 'incomplete' },
    spawn() { assert.fail('must not launch with an incomplete identity') }
  })
  assert.equal(result.status, 1)
})

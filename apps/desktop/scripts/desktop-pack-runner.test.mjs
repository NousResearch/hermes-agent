import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { runDesktopBuilder } from './desktop-pack-runner.mjs'
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

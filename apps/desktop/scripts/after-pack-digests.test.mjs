import assert from 'node:assert/strict'
import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { test } from 'vitest'

const repo = path.resolve(import.meta.dirname, '../../..')
const hook = pathToFileURL(path.join(import.meta.dirname, 'after-pack.mjs')).href
const python = process.env.HERMES_PYTHON || process.env.UV_PYTHON || 'python'

function fixture() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'after-pack-digests-'))
  const payload = path.join(directory, 'resources', 'agent-payload')
  const binary = path.join(payload, 'tools', 'uv', 'uv.exe')
  fs.mkdirSync(path.dirname(binary), { recursive: true })
  // A truncated PE certificate table is a sanitizer input, not an executable.
  const pe = Buffer.alloc(392)
  pe.write('MZ')
  pe.writeUInt32LE(128, 0x3c)
  pe.writeUInt32LE(0x4550, 128)
  pe.writeUInt16LE(240, 148)
  pe.writeUInt16LE(0x20b, 152)
  pe.writeUInt32LE(16, 260)
  pe.writeUInt32LE(pe.length, 296)
  pe.writeUInt32LE(24, 300)
  fs.writeFileSync(binary, pe)
  const env = { ...process.env, HERMES_HOME: path.join(directory, 'home'),
    HERMES_RUNTIME_DIR: path.join(directory, 'runtime'),
    UV_PYTHON: python, UV_PYTHON_DOWNLOADS: 'never', UV_OFFLINE: '1',
    UV_CACHE_DIR: path.join(directory, 'uv-cache') }
  for (const key of Object.keys(env)) {
    if (key.startsWith('AZURE_SIGN_')) delete env[key]
  }
  delete env.PYTHONHOME
  delete env.PYTHONPATH
  // The digest hook only needs the manifest to mark the payload as present.
  fs.writeFileSync(path.join(payload, 'manifest.json'), '{}')
  execFileSync(python, ['-c',
    'from pathlib import Path; import sys; from scripts.bundles.payload import record_tools; from pm.store import current_target; root=Path(sys.argv[1]); record_tools(root,Path("pm/lock.json"),current_target(),{"uv":"uv"})',
    payload], { cwd: repo, env, encoding: 'utf8' })
  const facts = path.join(payload, 'tools', 'facts.json')
  const before = JSON.parse(fs.readFileSync(facts, 'utf8'))
  const context = { electronPlatformName: process.platform, appOutDir: directory,
    packager: { appInfo: { productFilename: 'Hermes' }, config: {}, buildResourcesDir: directory } }
  const run = () => spawnSync(process.execPath, ['--input-type=module', '-e',
    `import afterPack from ${JSON.stringify(hook)}; await afterPack(${JSON.stringify(context)})`,
    'after-pack-test'], { cwd: directory, env, encoding: 'utf8', timeout: 60000 })
  return { directory, payload, binary, facts, before, run, env }
}

test.runIf(process.platform === 'win32')('afterPack records actual sanitized payload bytes', () => {
  const f = fixture()
  try {
    const result = f.run()
    assert.equal(result.status, 0, result.stdout + result.stderr)
    const finalBytes = fs.readFileSync(f.binary)
    assert.equal(finalBytes.readUInt32LE(296), 0)
    assert.equal(finalBytes.readUInt32LE(300), 0)
    const digest = execFileSync(python, ['-c',
      'from pathlib import Path; import sys; from pm.store import tree_digest; print(tree_digest(Path(sys.argv[1])))',
      path.dirname(f.binary)], { cwd: repo, env: f.env, encoding: 'utf8' }).trim()
    const after = JSON.parse(fs.readFileSync(f.facts, 'utf8'))
    assert.notEqual(digest, f.before.packages.uv.digest)
    assert.deepEqual(after.packages.uv, { ...f.before.packages.uv, digest })
  } finally {
    fs.rmSync(f.directory, { recursive: true, force: true })
  }
})

test.runIf(process.platform === 'win32')('afterPack refuses missing or corrupt facts only for a present payload', () => {
  const f = fixture()
  try {
    for (const contents of ['not-json', null]) {
      if (contents === null) fs.unlinkSync(f.facts)
      else fs.writeFileSync(f.facts, contents)
      const result = f.run()
      assert.notEqual(result.status, 0, result.stdout + result.stderr)
      if (contents !== null) assert.equal(fs.readFileSync(f.facts, 'utf8'), contents)
      else assert.equal(fs.existsSync(f.facts), false)
    }
    fs.unlinkSync(path.join(f.payload, 'manifest.json'))
    const light = f.run()
    assert.equal(light.status, 0, light.stdout + light.stderr)
    assert.equal(fs.existsSync(f.facts), false)
  } finally {
    fs.rmSync(f.directory, { recursive: true, force: true })
  }
})

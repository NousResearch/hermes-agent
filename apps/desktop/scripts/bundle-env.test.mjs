import { build } from 'esbuild'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test } from 'vitest'

import { applyBundleEnvironment, environmentDefaultsBanner } from './bundle-env.mjs'

test('explicit clears beat inherited homes and prevent Windows registry fallback before spawning', async () => {
  const root = mkdtempSync(join(tmpdir(), 'rabbit-bundle-clear-'))
  const paths = fileURLToPath(new URL('../electron/data-paths.ts', import.meta.url))
  const defaults = { RABBIT_HOME: null, RABBIT_DATA_DIR_SUFFIX: 'magic-test' }
  try {
    const entry = join(root, 'entry.mjs')
    writeFileSync(entry, `
      import {resolveDesktopRabbitHome} from ${JSON.stringify(paths)};
      import {execFileSync} from 'node:child_process';
      let registryReads = 0;
      const home = resolveDesktopRabbitHome({
        home: 'C:/Users/test', platform: 'win32', env: process.env,
        readWindowsHome: () => { registryReads++; return 'C:/old-rabbit'; }
      });
      console.log(JSON.stringify({cleared: process.env.RABBIT_HOME, home, registryReads,
        child: execFileSync(process.execPath, ['-p', 'process.env.RABBIT_HOME'], {
          env: {...process.env, RABBIT_HOME: home}, encoding: 'utf8'
        }).trim()}));
    `)
    const outfile = join(root, 'bundle.mjs')
    await build({ entryPoints: [entry], bundle: true, platform: 'node', format: 'esm', outfile,
      banner: { js: environmentDefaultsBanner(JSON.stringify(defaults)) } })
    const env = { ...process.env, RABBIT_HOME: 'C:/old-rabbit', LOCALAPPDATA: 'C:/Users/test/AppData/Local' }
    delete env.RABBIT_DATA_DIR_SUFFIX
    delete env.RABBIT_DESKTOP_USER_DATA_DIR
    const actual = JSON.parse(execFileSync(process.execPath, [outfile], { env, encoding: 'utf8' }))
    expect(actual).toEqual({ cleared: '', home: 'C:\\Users\\test\\AppData\\Local\\rabbitmagic-test',
      child: 'C:\\Users\\test\\AppData\\Local\\rabbitmagic-test', registryReads: 0 })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('applyBundleEnvironment replays the banner semantics for defaults, runtime overrides and clears', async () => {
  const root = mkdtempSync(join(tmpdir(), 'rabbit-bundle-pure-'))
  const defaults = { RABBIT_HOME: null, RABBIT_DATA_DIR_SUFFIX: 'baked', RABBIT_GUEST_ONBOARDING: '1' }
  const keys = Object.keys(defaults)
  const base = { RABBIT_HOME: 'runtime', RABBIT_DATA_DIR_SUFFIX: 'explicit' }
  try {
    // The bundled banner must produce the same effective environment the pure
    // function computes, so the smoke driver can predict the app's home from
    // the same bundle env data without running a child process.
    writeFileSync(join(root, 'reader.mjs'), `export const values = Object.fromEntries(${JSON.stringify(keys)}.map(key => [key, process.env[key]]));`, 'utf8')
    const entry = join(root, 'entry.mjs')
    writeFileSync(entry, `import {values} from './reader.mjs'; console.log(JSON.stringify(values));`, 'utf8')
    const outfile = join(root, 'bundle.mjs')
    await build({ entryPoints: [entry], bundle: true, platform: 'node', format: 'esm', outfile, banner: { js: environmentDefaultsBanner(JSON.stringify(defaults)) } })
    const env = { ...process.env, ...base }
    delete env.RABBIT_GUEST_ONBOARDING
    const viaBanner = JSON.parse(execFileSync(process.execPath, [outfile], { env, encoding: 'utf8' }))
    const viaFunction = Object.fromEntries(keys.map(key => [key, applyBundleEnvironment(base, defaults)[key]]))
    expect(viaFunction).toEqual(viaBanner)
    expect(viaFunction).toEqual({
      RABBIT_HOME: '', RABBIT_DATA_DIR_SUFFIX: 'explicit', RABBIT_GUEST_ONBOARDING: '1',
    })
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('baked defaults precede imported module initialization and reach children without overriding explicit env', async () => {
  const root = mkdtempSync(join(tmpdir(), 'rabbit-bundle-env-'))
  const defaults = { RABBIT_GUEST_ONBOARDING: '1', RABBIT_DATA_DIR_SUFFIX: 'magic-test', RABBIT_SHARED_AUTH_DIR: 'a=b "q"\n$(no)', RABBIT_SKIP_INTRO: '' }
  const env = { ...process.env }
  for (const key of Object.keys(defaults)) {
    delete env[key]
  }
  try {
    writeFileSync(join(root, 'reader.mjs'), `export const values = Object.fromEntries(${JSON.stringify(Object.keys(defaults))}.map(key => [key, process.env[key]]));`, 'utf8')
    const entry = join(root, 'entry.mjs')
    writeFileSync(entry, `import {values} from './reader.mjs'; import {execFileSync} from 'node:child_process'; console.log(JSON.stringify({values, child: execFileSync(process.execPath, ['-p', 'process.env.RABBIT_DATA_DIR_SUFFIX'], {encoding:'utf8'}).trim()}));`, 'utf8')
    const outfile = join(root, 'bundle.mjs')
    await build({ entryPoints: [entry], bundle: true, platform: 'node', format: 'esm', outfile, banner: { js: environmentDefaultsBanner(JSON.stringify(defaults)) } })
    const run = extra => JSON.parse(execFileSync(process.execPath, [outfile], { env: { ...env, ...extra }, encoding: 'utf8' }))
    expect(run({})).toEqual({ values: defaults, child: defaults.RABBIT_DATA_DIR_SUFFIX })
    expect(run({ RABBIT_DATA_DIR_SUFFIX: '-explicit', RABBIT_GUEST_ONBOARDING: '' })).toEqual({ values: { ...defaults, RABBIT_DATA_DIR_SUFFIX: '-explicit', RABBIT_GUEST_ONBOARDING: '' }, child: '-explicit' })
    for (const bad of ['[]', 'null', '{"BAD-NAME":"x"}', '{"RABBIT_HOME":1}', '{"RABBIT_HOME":"\\u0000"}',
      '{"NODE_OPTIONS":"--require=evil"}', '{"PATH":null}', '{"RABBIT_PYTHON":"/untrusted/python"}']) {
      expect(() => environmentDefaultsBanner(bad)).toThrow()
    }
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

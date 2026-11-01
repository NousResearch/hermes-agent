import { spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join } from 'node:path'
import { afterEach, expect, test } from 'vitest'

// Adapted from kvnloo's published real-tool evidence (b484599,
// whitespace-hooks.test.mjs: RED 0/2 on 88422bc186 with Lefthook 2.1.14, Linux
// only). saralilyb's #135279 diagnosis: Lefthook renames a user hook to .old
// when core.hooksPath comes from an included config. A set key is custom even
// when its value is one space (a valid directory) or explicitly empty.
const prepare = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).scripts.prepare
// The workspace-installed Lefthook. Resolution throws, failing the test, when it is missing.
const lefthookBin = join(dirname(dirname(createRequire(import.meta.url).resolve('lefthook/package.json'))), '.bin')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true, maxRetries: 3 }) })

function run(command, args, cwd, env) {
  const result = spawnSync(command, args, { cwd, env, encoding: 'utf8', timeout: 30000 })
  expect(result.error, `${command}: ${result.stderr}`).toBeUndefined()
  expect(result.signal).toBeNull()
  expect(result.status, `${command} ${args.join(' ')}: ${result.stderr}`).toBe(0)
  return result.stdout
}

// Real Git, npm and Lefthook: a missing tool fails rather than skips. The
// sentinel hook is a POSIX script and Windows rejects a one-space directory.
for (const value of [' ', '']) {
  test.skipIf(process.platform === 'win32')(`included core.hooksPath ${JSON.stringify(value)} skips lefthook and preserves hooks and config`, () => {
    const root = mkdtempSync(join(tmpdir(), 'hooks presence '))
    roots.push(root)
    const home = join(root, 'home')
    const cwd = join(root, 'repository')
    mkdirSync(home)
    mkdirSync(cwd)
    // Allowlist: never inherit Git config overrides, npm settings or Hermes secrets.
    const env = {
      PATH: `${lefthookBin}${delimiter}${process.env.PATH ?? ''}`, HOME: home,
      USERPROFILE: home, XDG_CONFIG_HOME: join(home, '.config'), GIT_CONFIG_NOSYSTEM: '1',
      GIT_CONFIG_GLOBAL: join(home, '.gitconfig'), GIT_TERMINAL_PROMPT: '0',
      npm_config_cache: join(root, 'npm-cache'), npm_config_userconfig: join(home, '.npmrc'),
      npm_config_globalconfig: join(root, 'npm-globalrc'), npm_config_audit: 'false',
      npm_config_fund: 'false', npm_config_update_notifier: 'false'
    }
    run('git', ['init', '--quiet'], cwd, env)
    run('lefthook', ['version'], cwd, env)
    const included = join(home, 'included.gitconfig')
    run('git', ['config', '--file', included, 'core.hooksPath', value], cwd, env)
    run('git', ['config', '--global', 'include.path', included], cwd, env)
    // The production query: the key is present (status 0) with this exact value.
    expect(run('git', ['config', '--get', 'core.hooksPath'], cwd, env)).toBe(`${value}\n`)
    const hooks = join(cwd, value)
    mkdirSync(hooks, { recursive: true })
    const sentinel = join(hooks, 'pre-commit')
    writeFileSync(sentinel, '#!/bin/sh\n# user-owned sentinel\nexit 0\n', { mode: 0o755 })
    writeFileSync(join(cwd, 'package.json'), JSON.stringify({
      name: 'hooks-fixture',
      version: '1.0.0', private: true, scripts: { prepare }
    }))
    writeFileSync(join(cwd, 'lefthook.yml'), 'pre-commit:\n  commands:\n    fixture:\n      run: "echo fixture"\n')
    const files = [sentinel, included, join(home, '.gitconfig'), join(cwd, '.git/config')]
    const before = files.map(path => readFileSync(path))
    const entries = readdirSync(hooks)
    const output = run('npm', ['run', 'prepare'], cwd, env)
    for (let i = 0; i < files.length; i++) expect(readFileSync(files[i]), `${files[i]} changed`).toEqual(before[i])
    expect(readdirSync(hooks), 'no backup or additional hook').toEqual(entries)
    // Empty is present too: the skip is explicit, not an accidental Lefthook failure.
    expect(output).toMatch(/^prepare: custom core.hooksPath set, skipping lefthook install$/m)
  }, 60000)
}

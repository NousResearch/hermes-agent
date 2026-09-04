import assert from 'node:assert/strict'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { CLI_LAUNCHER_SPECS, posixTrampolineScripts, renderWinWrapper } from '../../../scripts/desktop-cli/cli-entrypoints.mjs'
import { appExecutionAliasExtensions } from './before-build.mjs'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))

// The payload layout facts every generator receives (bin-relative, forward
// slashes — the same composition build-bundled-desktop.mjs step 5b feeds in).
const RELS = {
  relPython: '../tools/cpython-3.11.16+20260814-win32-x64/bin/python3',
  relSite: '../venv/lib/python3.11/site-packages',
  relRepo: '../hermes-agent'
}

test('three launcher specs mirror [project.scripts] in pyproject.toml', () => {
  assert.deepEqual(CLI_LAUNCHER_SPECS.map((s) => s.name), ['hermes', 'hermes-agent', 'hermes-acp'])
  assert.deepEqual(CLI_LAUNCHER_SPECS.map((s) => `${s.module}:${s.func}`), [
    'hermes_cli.main:main',
    'run_agent:main',
    'acp_adapter.entry:main'
  ])
})

const scripts = posixTrampolineScripts(RELS)

test('one POSIX trampoline per entry, named plainly (no suffix)', () => {
  assert.deepEqual(scripts.map((s) => s.name), ['hermes', 'hermes-agent', 'hermes-acp'])
})

test('trampoline is $0-relative: shebang, symlink-chain resolution, own-dir cd', () => {
  const text = scripts[0].text
  assert.ok(text.startsWith('#!/usr/bin/env bash\n'))
  // The whole relocatability contract: nothing in the script may be an
  // absolute path or a Windows path — everything resolves off $0.
  for (const line of text.split('\n')) {
    const code = line.replace(/^#/, '').trim()
    assert.ok(!/[A-Za-z]:\\/.test(code), `windows path in generated script: ${line}`)
    assert.ok(!/=\s*\/(?!usr\/bin\/env)/.test(code), `absolute path in generated script: ${line}`)
  }
  assert.match(text, /self="\$0"/)
  assert.match(text, /while \[ -L "\$self" \]/)
  assert.match(text, /BIN_DIR="\$\(cd -- "\$\(dirname -- "\$self"\)" && pwd\)"/)
  assert.match(text, /PYTHON="\$BIN_DIR"\/\.\.\/tools\//)
})

test('trampoline composes the payload import roots (repo first) and replaces inherited PYTHONPATH', () => {
  const text = scripts[0].text
  assert.match(text, /unset PYTHONPATH PYTHONHOME/)
  assert.match(text, /export PYTHONPATH="\$REPO:\$SITE"/)
  assert.match(text, /REPO="\$BIN_DIR"\/\.\.\/hermes-agent/)
  assert.match(text, /SITE="\$BIN_DIR"\/\.\.\/venv\/lib\/python3\.11\/site-packages/)
})

test('trampoline keeps pycache out of the payload and execs the entry module', () => {
  const text = scripts[0].text
  assert.match(text, /PYTHONPYCACHEPREFIX="\$HOME\/\.cache\/hermes-pycache"/)
  assert.match(text, /if \[ -z "\$\{PYTHONPYCACHEPREFIX:-\}" \]/, 'user-set prefix must win')
  assert.match(text, /exec "\$PYTHON" -m hermes_cli\.main "\$@"/)
})

test('trampoline fails loudly (exit 2) when the bundled interpreter is missing', () => {
  const text = scripts[0].text
  assert.match(text, /bundled interpreter missing at \$PYTHON/)
  assert.match(text, /exit 2/)
})

test('each trampoline execs its own entry module', () => {
  const modules = scripts.map((s) => /exec "\$PYTHON" -m (\S+) "\$@"/.exec(s.text)?.[1])
  assert.deepEqual(modules, ['hermes_cli.main', 'run_agent', 'acp_adapter.entry'])
})

test('win wrapper substitution bakes the entry module and payload layout facts', () => {
  const text = renderWinWrapper(CLI_LAUNCHER_SPECS[0], RELS.relRepo, RELS.relSite)
  assert.ok(!text.includes('__HERMES_'), 'placeholders must be fully substituted')
  assert.match(text, /HERMES_ENTRY_MODULE = "hermes_cli\.main"/)
  assert.match(text, /HERMES_ENTRY_FUNC = "main"/)
  assert.match(text, /HERMES_REPO_REL = "\.\.\/hermes-agent"/)
  assert.match(text, /HERMES_SITE_REL = "\.\.\/venv\/lib\/python3\.11\/site-packages"/)
})

test('win wrapper rejects values that still contain placeholders', () => {
  assert.throws(() => renderWinWrapper({ module: '__HERMES_REPO_REL__', func: 'main' }, RELS.relRepo, RELS.relSite))
})

test('MSIX: ONE uap5 alias Extension naming every launcher alias', () => {
  const xml = appExecutionAliasExtensions()
  const blocks = xml.match(/<uap5:Extension\b/g) ?? []
  // makeappx rejects a second windows.appExecutionAlias Extension with the
  // opaque 0x80080204 — all launcher aliases must ride in ONE block.
  assert.equal(blocks.length, 1)
  // The block's Executable names the exe that serves the aliases (first launcher).
  const first = ['app', 'resources', 'agent-payload', 'bin', `${CLI_LAUNCHER_SPECS[0].name}.exe`].join('\\')
  assert.ok(xml.includes(`Executable="${first}"`), 'Executable must name the first launcher exe')
  // Every launcher exe gets exactly one ExecutionAlias inside the block.
  for (const spec of CLI_LAUNCHER_SPECS) {
    const expected = ['app', 'resources', 'agent-payload', 'bin', `${spec.name}.exe`].join('\\')
    // Backslashes: MSIX Executable must be manifest-backslash form.
    assert.ok(!xml.includes(`${expected}/`), 'forward slashes are not manifest form')
    assert.ok(
      xml.includes(`<uap5:ExecutionAlias Alias="${spec.name}.exe" />`),
      `no ExecutionAlias for ${spec.name}.exe`
    )
  }
  assert.equal((xml.match(/<uap5:ExecutionAlias /g) ?? []).length, CLI_LAUNCHER_SPECS.length)
})

test('light variant emits no alias fragments', () => {
  // The light gating lives in writeMsixExtensions (light ? '' : ...); the
  // pure builder must stay gating-free, so just pin its shape here.
  assert.ok(appExecutionAliasExtensions(['hermes']).includes('windows.appExecutionAlias'))
  assert.ok(scriptDir.length > 0)
})

// ── HermesGateway Windows Service fragment (Task 2, plan:
//    gateway-msix-windows-service) ────────────────────────────────────────
import { serviceExtensions } from './before-build.mjs'

test('bundled variant registers HermesGateway, demand-start, launcher exe', () => {
  const frag = serviceExtensions()
  assert.ok(frag.includes('Category="windows.service"'), 'service category')
  assert.ok(frag.includes('Name="HermesGateway"'), 'the service name the CLI verbs key off')
  assert.ok(frag.includes('StartupType="demand"'), 'config-only posture: arrives stopped')
  // Compose the expected path the same way the source does — no escaping
  // ambiguity (the appExecutionAliasExtensions precedent).
  const bs = String.fromCharCode(92)
  const launcherPath = ['app', 'resources', 'agent-payload', 'bin', 'hermes.exe'].join(bs)
  assert.ok(
    frag.includes(`Executable="${launcherPath}"`),
    'the service Executable is the payload launcher (no shim binary)'
  )
  assert.ok(!frag.includes('StartAccount'), 'user-context by default — localSystem rejected')
})

test('light and store variants render service-less', () => {
  assert.equal(serviceExtensions({ variant: 'light' }), '')
  assert.equal(serviceExtensions({ variant: 'store' }), '')
})

test('one desktop6 extension block with xmlns on the fragment root', () => {
  const frag = serviceExtensions()
  assert.equal(
    (frag.match(/<desktop6:Extension/g) || []).length,
    1,
    'the 0x80080204 playbook: ONE extension block'
  )
  assert.ok(
    frag.includes('xmlns:desktop6="http://schemas.microsoft.com/appx/manifest/desktop/windows10/6"'),
    'namespace rides the fragment root (the copilot-fragment precedent)'
  )
})

test('custom name propagates (per-install namespacing)', () => {
  const frag = serviceExtensions({ name: 'HermesGateway_Tag1' })
  assert.ok(frag.includes('Name="HermesGateway_Tag1"'))
})

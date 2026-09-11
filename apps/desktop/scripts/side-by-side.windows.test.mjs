// Native SDK proof, NOT desktop E2E. Run on Windows with desktop dependencies:
//   node apps/desktop/scripts/side-by-side.windows.test.mjs
// makeappx validates unmodified production manifests in a temporary tree.
// where.exe and fixture PNGs stand in for payload/artwork. No package is
// registered; no applications, profile data, certificates or policy are changed.
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const { test } = await import(process.env.VITEST ? 'vitest' : 'node:test')

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', windowsHide: true, ...options })
  if (result.error) throw result.error
  return { ...result, output: `${result.stdout || ''}${result.stderr || ''}` }
}

function checked(command, args, options) {
  const result = run(command, args, options)
  assert.equal(result.status, 0, `${command} ${args.join(' ')}\n${result.output}`)
  return result.stdout.trim()
}

function attribute(xml, element, name) {
  const tag = xml.match(new RegExp(`<${element}\\b[^>]*>`))?.[0]
  const value = tag?.match(new RegExp(`\\b${name}=["']([^"']*)["']`))?.[1]
  assert.ok(value, `Missing ${element}.${name}`)
  return value
}

async function nativeProof() {
  const sdkRoot = path.join(process.env['ProgramFiles(x86)'], 'Windows Kits/10/bin')
  const sdkArch = process.arch === 'arm64' ? 'arm64' : 'x64'
  const sdk = fs.readdirSync(sdkRoot).sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))
    .map(version => path.join(sdkRoot, version, sdkArch, 'makeappx.exe')).find(file => fs.existsSync(file))
  assert.ok(sdk, `Windows SDK makeappx.exe (${sdkArch}) is required`)
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sxs-sdk-'))
  // Retain failed runs for native diagnostics. Successful SDK-only runs leave
  // a small receipt too; neither branch touches the source checkout's build/.
  console.log(`Native Windows fixture proof: ${root}`)
  const work = path.join(root, 'source')
  const desktop = path.join(work, 'apps/desktop')
  const copied = [
    'apps/desktop/product-identity.cjs', 'apps/desktop/electron-builder.config.cjs',
    'apps/desktop/package.json', 'apps/desktop/update-feed.cjs', 'apps/desktop/update-feed.json',
    'apps/desktop/assets/msix-manifest.xml',
    ...['before-build', 'gen-msix-manifest', 'gen-appinstaller', 'mac-sign', 'payload-digests', 'write-build-stamp', 'utils']
      .map(name => `apps/desktop/scripts/${name}.mjs`),
    'scripts/msix-shared.mjs', 'scripts/release-content-types.json', 'scripts/build/python.mjs',
  ]
  for (const file of copied) {
    fs.mkdirSync(path.dirname(path.join(work, file)), { recursive: true })
    fs.cpSync(path.join(repo, file), path.join(work, file), { recursive: true })
  }
  // The fixture does not test artwork. Supply valid PNGs to the real staging
  // hook without requiring the (separate) icon-generation build prerequisite.
  const iconsScript = path.join(root, 'fixture-icons.ps1')
  fs.writeFileSync(iconsScript, String.raw`
param([string]$Dir)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $Dir | Out-Null
foreach ($asset in @(@('Square44x44Logo.png',44,44), @('Square150x150Logo.png',150,150), @('StoreLogo.png',50,50), @('Wide310x150Logo.png',310,150))) {
  $image = New-Object System.Drawing.Bitmap([int]$asset[1], [int]$asset[2])
  try { $image.Save((Join-Path $Dir $asset[0]), [System.Drawing.Imaging.ImageFormat]::Png) }
  finally { $image.Dispose() }
}
`)
  checked('powershell.exe', ['-NoProfile', '-NonInteractive', '-File', iconsScript, '-Dir', path.join(desktop, 'assets/appx')])
  fs.symlinkSync(path.join(repo, 'node_modules'), path.join(work, 'node_modules'), 'junction')
  const env = { ...process.env, HERMES_HOME: path.join(root, 'home'), HERMES_RUNTIME_DIR: path.join(root, 'runtime') }
  for (const key of ['HERMES_BUILD_COMMIT', 'HERMES_PAYLOAD_TAG', 'HERMES_PAYLOAD_VERSION', 'HERMES_DESKTOP_VARIANT', 'BUILD_NUMBER']) delete env[key]
  const gitEnv = { ...env, GIT_AUTHOR_NAME: 'Native fixture', GIT_AUTHOR_EMAIL: 'fixture@example.invalid', GIT_COMMITTER_NAME: 'Native fixture', GIT_COMMITTER_EMAIL: 'fixture@example.invalid', GIT_AUTHOR_DATE: '2026-09-01T00:00:00Z', GIT_COMMITTER_DATE: '2026-09-01T00:00:00Z' }
  const git = (...args) => checked('git', args, { cwd: work, env: gitEnv })
  git('init', '-q')
  git('-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'Stable fixture')
  git('tag', 'v1.2.3')
  const commitA = git('rev-parse', 'HEAD')
  git('-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'Second commit fixture')
  const commitB = git('rev-parse', 'HEAD')
  assert.notEqual(commitA, commitB)
  const cases = [
    ['stable', { HERMES_PAYLOAD_TAG: 'v1.2.3' }],
    ['canary', { HERMES_PAYLOAD_TAG: 'v1.2.4-canary.20260902000000' }],
    ['canary-update', { HERMES_PAYLOAD_TAG: 'v1.2.4-canary.20260903000000' }],
    ['commit-a', { HERMES_BUILD_COMMIT: commitA, HERMES_PAYLOAD_VERSION: '1.2.4' }],
    ['commit-b', { HERMES_BUILD_COMMIT: commitB, HERMES_PAYLOAD_VERSION: '1.2.4' }],
  ]
  const rows = []
  const failures = []
  const check = (value, message) => { if (!value) failures.push(message) }
  for (const [label, flavorEnv] of cases) {
    const childEnv = { ...env, HERMES_DESKTOP_VARIANT: 'bundled', ...flavorEnv }
    const node = args => checked(process.execPath, args, { cwd: work, env: childEnv })
    const packageDir = path.join(root, label)
    fs.mkdirSync(packageDir)
    const manifestDir = path.join(desktop, 'build/agent-payload')
    fs.rmSync(manifestDir, { recursive: true, force: true })
    fs.mkdirSync(path.join(manifestDir, 'bin'), { recursive: true })
    fs.copyFileSync(path.join(process.env.SystemRoot, 'System32/where.exe'), path.join(manifestDir, 'bin/hermes.exe'))
    fs.writeFileSync(path.join(manifestDir, 'manifest.json'), JSON.stringify({
      target: `win32-${process.arch}`, launchers: ['hermes'], runtime: { commands: { hermes: 'bin/hermes.exe' } },
    }))
    // Fresh processes match the per-build module cache boundary. No fabricated
    // product identity/config; these are the production hook and generators.
    const moduleUrl = relative => pathToFileURL(path.join(work, relative)).href
    const facts = JSON.parse(node(['--input-type=module', '-e', `
      import beforeBuild from ${JSON.stringify(moduleUrl('apps/desktop/scripts/before-build.mjs'))};
      import identity from ${JSON.stringify(moduleUrl('apps/desktop/product-identity.cjs'))};
      import config from ${JSON.stringify(moduleUrl('apps/desktop/electron-builder.config.cjs'))};
      import { appIdentity } from ${JSON.stringify(moduleUrl('scripts/msix-shared.mjs'))};
      import { stageDesktopLaunchers } from ${JSON.stringify(moduleUrl('apps/desktop/scripts/write-build-stamp.mjs'))};
      import { AppInfo } from ${JSON.stringify(moduleUrl('node_modules/app-builder-lib/dist/appInfo.js'))};
      import fs from 'node:fs';
      const metadata = { ...JSON.parse(fs.readFileSync(${JSON.stringify(path.join(desktop, 'package.json'))}, 'utf8')), ...config.extraMetadata };
      const executable = new AppInfo({ config, metadata }, null, config.win).productFilename + '.exe';
      const payload = stageDesktopLaunchers(${JSON.stringify(manifestDir)});
      await beforeBuild();
      console.log(JSON.stringify({identity, config, payload, executable, app: appIdentity(${JSON.stringify(desktop)})}));
    `, path.join(root, 'native-probe.mjs')]))
    const xml = node([path.join(desktop, 'scripts/gen-msix-manifest.mjs'), 'bundled', process.arch])
    fs.writeFileSync(path.join(packageDir, 'AppxManifest.xml'), xml)
    fs.cpSync(path.join(desktop, 'build/appx'), path.join(packageDir, 'assets'), { recursive: true })
    fs.mkdirSync(path.join(packageDir, 'Public'))
    fs.cpSync(manifestDir, path.join(packageDir, 'app/resources/agent-payload'), { recursive: true })
    fs.copyFileSync(path.join(process.env.SystemRoot, 'System32/where.exe'), path.join(packageDir, 'app', facts.executable))
    const expectedExecutable = path.win32.join('app/resources/agent-payload', facts.payload.runtime.commands.hermes)
    check(attribute(xml, 'Application', 'Executable') === path.win32.join('app', facts.executable), `${label}: application does not launch its GUI executable`)
    const aliasExtension = /<uap5:Extension\b[^>]*Category="windows.appExecutionAlias"[^>]*>/.exec(xml)?.[0]
    check(aliasExtension && attribute(aliasExtension, 'uap5:Extension', 'Executable') === expectedExecutable, `${label}: CLI alias does not launch its payload executable`)
    for (const executable of new Set([...xml.matchAll(/\bExecutable="([^"]+)"/g)].map(match => match[1]))) {
      assert.ok(fs.existsSync(path.join(packageDir, executable)), `${label}: generated executable ${executable} is absent after launcher staging`)
    }
    const packed = path.join(root, `${label}.msix`)
    const pack = run(sdk, ['pack', '/d', packageDir, '/p', packed, '/o'])
    fs.writeFileSync(path.join(root, `${label}-makeappx.log`), pack.output)
    assert.equal(pack.status, 0, `${label}: SDK rejected generated manifest\n${pack.output}`)
    assert.ok(fs.statSync(packed).size > 0)
    const unpacked = path.join(root, `${label}-unpacked`)
    checked(sdk, ['unpack', '/p', packed, '/d', unpacked, '/o'])
    const roundtrip = fs.readFileSync(path.join(unpacked, 'AppxManifest.xml'), 'utf8')
    const name = attribute(roundtrip, 'Identity', 'Name')
    const version = attribute(roundtrip, 'Identity', 'Version')
    const aliases = [...roundtrip.matchAll(/<uap5:ExecutionAlias\s+Alias="([^"]+)"/g)].map(match => match[1])
    check(name === facts.identity.msixAppIdWithOrg, `${label}: manifest identity differs from generated product`)
    check(version === facts.app.version, `${label}: manifest version differs from appIdentity`)
    check(aliases.length === 1 && aliases[0] === `${facts.identity.cliName}.exe`, `${label}: aliases ${aliases} do not match ${facts.identity.cliName}`)
    const descriptor = path.join(root, `${label}.appinstaller`)
    const generated = run(process.execPath, [path.join(desktop, 'scripts/gen-appinstaller.mjs'), '--out', descriptor, '--base-url', 'https://example.invalid/fixture'], { cwd: work, env: childEnv })
    if (flavorEnv.HERMES_BUILD_COMMIT) {
      check(generated.status !== 0 && !fs.existsSync(descriptor), `${label}: commit build emitted App Installer feed`)
      check(facts.config.publish === null, `${label}: commit build config still publishes`)
    } else {
      assert.equal(generated.status, 0, generated.output)
      check(attribute(fs.readFileSync(descriptor, 'utf8'), 'MainBundle', 'Name') === name, `${label}: App Installer targets another family`)
    }
    const row = { label, name, version, aliases, packageDir, cliName: facts.identity.cliName, commit: flavorEnv.HERMES_BUILD_COMMIT || null }
    rows.push(row)
    console.log(`SDK pack/unpack PASS ${label}: ${name} ${version}; aliases=${aliases.join(',')}`)
  }
  const [stable, canary, update, a, b] = rows
  check(new Set([stable, canary, a, b].map(row => row.name)).size === 4, 'Flavor package identities collide')
  check(new Set([stable, canary, a, b].flatMap(row => row.aliases)).size === 4, 'Flavor execution aliases collide')
  check(canary.name === update.name && canary.name !== stable.name, 'Canary upgrade does not stay in its own family')
  check(canary.version.localeCompare(update.version, undefined, { numeric: true }) < 0, 'Canary version does not increase')
  fs.writeFileSync(path.join(root, 'rows.json'), JSON.stringify(rows, null, 2))
  fs.writeFileSync(path.join(root, 'contracts.json'), JSON.stringify({ failures }, null, 2))
  assert.deepEqual(failures, [], 'Generated packaging contract failures')
  console.log(`Receipt: ${root} (fixture packaging only; no registration, app launch, signing, artwork or payload E2E)`)
}

const options = { skip: process.platform !== 'win32', timeout: 180_000 }
test('native Windows SDK packs distinct flavor identities and ordered canary updates', options, nativeProof)

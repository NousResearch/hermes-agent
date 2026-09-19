// Fresh-install admission is separate from the stricter OLD -> NEW update contract.
import fs from 'node:fs'
import path from 'node:path'
import { execFileSync, spawnSync } from 'node:child_process'
import { parseArgs } from 'node:util'
import { fileURLToPath } from 'node:url'
import { OUT_OF_STORE_PUBLISHER, channelBuildRequest } from '../../../scripts/msix-shared.mjs'
import { codesignTeam, channelStampAssertions } from './mac-bundled-manifest.cjs'

const identityModule = fileURLToPath(new URL('../../../apps/desktop/product-identity.cjs', import.meta.url))

function admitChannelRequest(request, commit, tag) {
  if (tag || request?.commit !== commit) throw new Error('Channel request conflicts with commit or tag')
  return channelBuildRequest({ ...process.env, HERMES_DESKTOP_VARIANT: 'bundled',
    HERMES_BUILD_COMMIT: '', HERMES_PAYLOAD_TAG: '', HERMES_PAYLOAD_VERSION: '',
    _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(request) })
}

export function bundleIdentity(commit, tag = '', channelRequest = null) {
  if (!/^[a-f0-9]{40}$/.test(commit)) throw new Error('Expected exact full lowercase commit SHA')
  if (tag && !/^v\d+\.\d+\.\d+(?:-canary\.20\d{6}(?:\d{6})?)?$/.test(tag)) throw new Error('Invalid release tag')
  if (channelRequest !== null) {
    const request = admitChannelRequest(channelRequest, commit, tag)
    return { appId: request.identity.appId, msixIdentity: request.identity.msixAppIdWithOrg,
      applicationId: request.identity.appNamePascal, publisher: OUT_OF_STORE_PUBLISHER,
      windowsVersion: request.windowsVersion }
  }
  // The production module reads build flags at require-time. A child avoids
  // mutating the driver's environment or returning a cached variant identity.
  const identity = JSON.parse(execFileSync(process.execPath, ['-e',
    'console.log(JSON.stringify(require(process.argv[1])))', identityModule], {
    encoding: 'utf8', env: { ...process.env, HERMES_DESKTOP_VARIANT: 'bundled',
      HERMES_PAYLOAD_TAG: tag, HERMES_BUILD_COMMIT: tag ? '' : commit, _HERMES_CHANNEL_REQUEST_JSON: '' },
  }))
  return { appId: identity.appId, msixIdentity: identity.msixAppIdWithOrg,
    applicationId: identity.appNamePascal, publisher: OUT_OF_STORE_PUBLISHER }
}

/** @param {object} stamp
 * @param {{commit: string, tag?: string, platform: string,
 * channelRequest?: import('../../../apps/desktop/electron/install-stamp.js').ChannelBuildRequest | null}} options */
export function verifyBundleStamp(stamp, { commit, tag = '', platform, channelRequest = null }) {
  if (!['darwin', 'win32'].includes(platform)) throw new Error('Unsupported native platform')
  const request = channelRequest === null ? null : admitChannelRequest(channelRequest, commit, tag)
  const expected = { commit, tag: request?.receiverCandidate ? request.releaseTag : tag || null, payload: 'bundled', distribution: 'desktop-app', dirty: false,
    updateMechanism: tag || request ? { darwin: 'electron-updater', win32: 'app-installer' }[platform] : 'external' }
  if (!tag && !request) Object.assign(expected, { source: 'commit-build', branch: null })
  for (const [key, value] of Object.entries(expected)) {
    if (stamp?.[key] !== value) throw new Error(`stamp.${key}: ${JSON.stringify(stamp?.[key])} != ${JSON.stringify(value)}`)
  }
  if (request) {
    const problems = channelStampAssertions(stamp, request)
    if (problems.length) throw new Error(problems.join('; '))
    return platform === 'darwin' ? request.version : request.windowsVersion
  }
  return legacyBundleVersion(stamp, tag)
}

function legacyBundleVersion(stamp, tag) {
  if (stamp.channelBuild != null) throw new Error('Unexpected stamp.channelBuild without an admitted request')
  const version = tag ? tag.slice(1) : stamp.displayVersion
  if (!tag && !/^\d+\.\d+\.\d+$/.test(version)) throw new Error('stamp.displayVersion must be a commit-build semver')
  return version
}

function childPath(root, relative) {
  const full = fs.realpathSync(path.resolve(root, relative))
  if (!full.startsWith(`${fs.realpathSync(root)}${path.sep}`)) throw new Error(`Path escapes installed root: ${relative}`)
  return full
}

export function verifyMacMetadata(plist, stamp, { commit, tag, channelRequest }) {
  const expected = bundleIdentity(commit, tag, channelRequest)
  if (plist.CFBundleIdentifier !== expected.appId) throw new Error('CFBundleIdentifier disagrees with product identity')
  const version = verifyBundleStamp(stamp, { commit, tag, platform: 'darwin', channelRequest })
  if (plist.CFBundleShortVersionString !== version) throw new Error('CFBundleShortVersionString disagrees with stamp/tag/request')
  if (channelRequest && plist.CFBundleVersion !== channelRequest.version) throw new Error('CFBundleVersion disagrees with channel request')
  if (!plist.CFBundleExecutable || /[/\\]/.test(plist.CFBundleExecutable)) throw new Error('Invalid CFBundleExecutable')
  return version
}

function verifyMac(app, commit, tag, arch, channelRequest) {
  const expected = bundleIdentity(commit, tag, channelRequest)

  execFileSync('/usr/bin/codesign', ['--verify', '--deep', '--strict', '--verbose=2', app], { stdio: 'inherit' })
  const display = spawnSync('/usr/bin/codesign', ['-dv', '--verbose=4', app], { encoding: 'utf8' })
  if (display.error || display.status !== 0) throw new Error(`codesign display failed: ${display.error || display.stderr}`)
  const signedTeam = codesignTeam(display.stderr)
  // The admitted artifact digest pins the signed bytes. Require a real team,
  // not a new CI identity setting that could drift from the signing certificate.
  if (!/^[A-Z0-9]{10}$/.test(signedTeam || '')) throw new Error('Bundle has no valid signing team')
  if (/^Identifier=(.+)$/m.exec(display.stderr)?.[1] !== expected.appId) throw new Error('Signing identifier disagrees with product identity')
  const plist = JSON.parse(execFileSync('/usr/bin/plutil', ['-convert', 'json', '-o', '-',
    path.join(app, 'Contents/Info.plist')], { encoding: 'utf8' }))
  const stamp = JSON.parse(fs.readFileSync(childPath(app, 'Contents/Resources/install-stamp.json'), 'utf8'))
  const version = verifyMacMetadata(plist, stamp, { commit, tag, channelRequest })
  const exe = childPath(app, `Contents/MacOS/${plist.CFBundleExecutable}`)
  fs.accessSync(exe, fs.constants.X_OK)
  const archs = execFileSync('/usr/bin/lipo', ['-archs', exe], { encoding: 'utf8' }).trim().split(/\s+/)
  const native = { arm64: 'arm64', x64: 'x86_64' }[arch]
  if (archs.length !== 1 || archs[0] !== native) throw new Error(`Expected native ${arch}, got ${archs}`)
  execFileSync('/usr/bin/xcrun', ['stapler', 'validate', app], { stdio: 'inherit' })
  execFileSync('/usr/sbin/spctl', ['-a', '-vv', '-t', 'exec', app], { stdio: 'inherit' })
  const root = childPath(app, 'Contents/Resources/agent-payload')
  return { app, exe, root, arch, version, bundleIdentifier: expected.appId, teamIdentifier: signedTeam, stamp }
}

function prepareDirectories(work, out) {
  const temp = fs.realpathSync(process.env.RUNNER_TEMP)
  const canonical = [work, out].map(dir => {
    if (!path.isAbsolute(dir)) throw new Error('Work and out must be absolute descendants of RUNNER_TEMP')
    dir = path.resolve(dir)
    let parent = dir
    while (!fs.existsSync(parent)) parent = path.dirname(parent)
    // macOS /var is itself an alias of /private/var; allow aliases only if
    // their actual destination is still beneath this runner's temporary root.
    const full = path.resolve(fs.realpathSync(parent), path.relative(parent, dir))
    const relative = path.relative(temp, full)
    if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      throw new Error('Work and out must stay beneath RUNNER_TEMP')
    }
    return full
  })
  if (canonical[0] === canonical[1] || canonical[0].startsWith(`${canonical[1]}${path.sep}`)) {
    throw new Error('Output must not contain the private work directory')
  }
  if (fs.existsSync(canonical[0])) throw new Error(`Refusing to reuse work directory: ${work}`)
  fs.mkdirSync(canonical[0], { recursive: true })
  fs.mkdirSync(canonical[1], { recursive: true })
}

function main() {
  const [command, ...args] = process.argv.slice(2)
  const { values } = parseArgs({ args, options: Object.fromEntries(
    ['commit', 'tag', 'platform', 'stamp', 'app', 'arch', 'out', 'work', 'channel-request'].map(key => [key, { type: 'string' }])) })
  if (values['channel-request'] !== undefined) {
    // Preserve the raw JSON for the protocol decoder's duplicate-key checks.
    const raw = fs.readFileSync(values['channel-request'], 'utf8')
    if (!raw) throw new Error('Empty channel request')
    values.channelRequest = channelBuildRequest({ ...process.env, HERMES_DESKTOP_VARIANT: 'bundled',
      HERMES_BUILD_COMMIT: '', HERMES_PAYLOAD_TAG: '', HERMES_PAYLOAD_VERSION: '', _HERMES_CHANNEL_REQUEST_JSON: raw })
  }
  const commands = {
    identity: () => bundleIdentity(values.commit, values.tag, values.channelRequest),
    stamp: () => verifyBundleStamp(JSON.parse(fs.readFileSync(values.stamp, 'utf8')), values),
    'verify-mac': () => verifyMac(values.app, values.commit, values.tag, values.arch, values.channelRequest),
    prepare: () => prepareDirectories(values.work, values.out),
  }
  if (!commands[command]) throw new Error(`Unknown command: ${command}`)
  const result = commands[command]()
  if (result !== undefined) {
    const text = JSON.stringify(result, null, 2) + '\n'
    if (values.out && command !== 'prepare') fs.writeFileSync(values.out, text)
    else process.stdout.write(text)
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()
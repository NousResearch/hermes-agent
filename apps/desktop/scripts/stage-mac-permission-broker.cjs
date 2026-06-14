'use strict'

const childProcess = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const BROKER_BUNDLE_ID = 'com.nousresearch.hermes.macbroker'
const BROKER_APP_NAME = 'HermesMacBroker.app'
const BROKER_EXECUTABLE_NAME = 'HermesMacBroker'
const BROKER_VERSION = '0.2.0'

function xmlEscape(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

function brokerSourcePath(desktopRoot) {
  return path.join(desktopRoot, 'macos', 'HermesMacBroker', 'Sources', 'HermesMacBroker', 'main.swift')
}

function brokerBuildRoot(desktopRoot) {
  return path.join(desktopRoot, 'build', 'mac-permission-broker')
}

function brokerBundlePath(root) {
  return path.join(root, BROKER_APP_NAME)
}

function brokerLoginItemsDir(appBundlePath) {
  return path.join(appBundlePath, 'Contents', 'Library', 'LoginItems')
}

function brokerBundleInfoPlist({ bundleId = BROKER_BUNDLE_ID, version = BROKER_VERSION } = {}) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>Hermes Mac Broker</string>
  <key>CFBundleExecutable</key>
  <string>${BROKER_EXECUTABLE_NAME}</string>
  <key>CFBundleIdentifier</key>
  <string>${xmlEscape(bundleId)}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>HermesMacBroker</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${xmlEscape(version)}</string>
  <key>CFBundleVersion</key>
  <string>${xmlEscape(version)}</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>LSUIElement</key>
  <true/>
  <key>NSAppleEventsUsageDescription</key>
  <string>Hermes uses Automation only for user-approved Mac app actions routed through the signed broker.</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Hermes uses the microphone for voice input when routed through the signed broker.</string>
  <key>NSUserNotificationsUsageDescription</key>
  <string>Hermes sends local notifications through the signed broker when requested.</string>
</dict>
</plist>
`
}

function swiftcArgs({ source, output }) {
  return [
    source,
    '-o',
    output,
    '-framework',
    'AppKit',
    '-framework',
    'ApplicationServices',
    '-framework',
    'AVFoundation',
    '-framework',
    'CoreGraphics',
    '-framework',
    'CryptoKit',
    '-framework',
    'ServiceManagement',
    '-framework',
    'UserNotifications'
  ]
}

function writeBrokerBundle({ desktopRoot, outputRoot = brokerBuildRoot(desktopRoot), execFile = childProcess.execFileSync } = {}) {
  if (!desktopRoot) throw new Error('desktopRoot is required')
  const source = brokerSourcePath(desktopRoot)
  if (!fs.existsSync(source)) throw new Error(`Missing HermesMacBroker Swift source: ${source}`)

  const bundle = brokerBundlePath(outputRoot)
  const contents = path.join(bundle, 'Contents')
  const macos = path.join(contents, 'MacOS')
  const infoPlist = path.join(contents, 'Info.plist')
  const executable = path.join(macos, BROKER_EXECUTABLE_NAME)

  fs.rmSync(bundle, { recursive: true, force: true })
  fs.mkdirSync(macos, { recursive: true })
  fs.writeFileSync(infoPlist, brokerBundleInfoPlist(), 'utf8')
  execFile('swiftc', swiftcArgs({ source, output: executable }), { stdio: 'pipe' })
  fs.chmodSync(executable, 0o755)
  return { bundle, executable, infoPlist }
}

function maybeAdHocSignBundle(bundle, { execFile = childProcess.execFileSync, enabled = false } = {}) {
  if (!enabled) return false
  execFile('/usr/bin/codesign', ['--force', '--deep', '--sign', '-', bundle], { stdio: 'pipe' })
  return true
}

function installBrokerIntoApp({ appBundlePath, brokerBundle, fsImpl = fs } = {}) {
  if (!appBundlePath) throw new Error('appBundlePath is required')
  if (!brokerBundle) throw new Error('brokerBundle is required')
  const loginItems = brokerLoginItemsDir(appBundlePath)
  const destination = path.join(loginItems, BROKER_APP_NAME)
  fsImpl.mkdirSync(loginItems, { recursive: true })
  fsImpl.rmSync(destination, { recursive: true, force: true })
  fsImpl.cpSync(brokerBundle, destination, { recursive: true })
  return destination
}

function shouldStageBroker(platform, env = process.env) {
  if (env.HERMES_DESKTOP_SKIP_MAC_BROKER === '1') return false
  return platform === 'darwin'
}

async function stageMacPermissionBroker(context, deps = {}) {
  const platform = context?.electronPlatformName
  if (!shouldStageBroker(platform, deps.env || process.env)) return null

  const desktopRoot = deps.desktopRoot || path.resolve(__dirname, '..')
  const productName = context.packager?.appInfo?.productFilename || 'Hermes'
  const appBundlePath = deps.appBundlePath || path.join(context.appOutDir, `${productName}.app`)
  const execFile = deps.execFile || childProcess.execFileSync
  const build = writeBrokerBundle({ desktopRoot, execFile })
  maybeAdHocSignBundle(build.bundle, {
    execFile,
    enabled: (deps.env || process.env).HERMES_DESKTOP_SIGN_MAC_BROKER_ADHOC === '1'
  })
  const installed = installBrokerIntoApp({ appBundlePath, brokerBundle: build.bundle })
  return { ...build, installed }
}

module.exports = {
  BROKER_APP_NAME,
  BROKER_BUNDLE_ID,
  BROKER_EXECUTABLE_NAME,
  BROKER_VERSION,
  brokerBuildRoot,
  brokerBundleInfoPlist,
  brokerBundlePath,
  brokerLoginItemsDir,
  brokerSourcePath,
  installBrokerIntoApp,
  maybeAdHocSignBundle,
  shouldStageBroker,
  stageMacPermissionBroker,
  swiftcArgs,
  writeBrokerBundle
}

if (require.main === module) {
  const desktopRoot = path.resolve(__dirname, '..')
  const outputRoot = process.argv[2] ? path.resolve(process.argv[2]) : path.join(os.tmpdir(), `hermes-macbroker-${process.pid}`)
  const build = writeBrokerBundle({ desktopRoot, outputRoot })
  maybeAdHocSignBundle(build.bundle, { enabled: process.env.HERMES_DESKTOP_SIGN_MAC_BROKER_ADHOC === '1' })
  console.log(JSON.stringify(build, null, 2))
}

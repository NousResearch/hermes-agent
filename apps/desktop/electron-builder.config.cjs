// THE electron-builder configuration — the whole thing, one file. There is
// no "build" field in package.json: run-electron-builder.mjs always passes
// --config for this file, so a stray package.json field would be silently
// ignored anyway, and splitting the config across JSON + this overlay is
// how the two halves drift.
//
// A .cjs module (not JSON) so the variant is decided at require time:
// HERMES_DESKTOP_VARIANT=light builds "Hermes Light". The whole config
// derives from that one flag.
// @ts-check
'use strict'

const fs = require('node:fs')
const path = require('node:path')

const {
  light,
  store,
  storeMsix,
  displayName,
  appId,
  appNamePascal,
  channel,
  msixAppIdWithOrg
} = require('./product-identity.cjs')

// `storeMsix` is optional on the identity type but guaranteed present when
// `store` is true (product-identity.cjs spreads it only in that branch).
// The `store` flag is typed `boolean` in the identity, so checkJs cannot
// correlate the two; `mustStoreMsix` is the single assertion point and the
// invariant lives in product-identity.cjs:33-34/58-68.
/** @type {NonNullable<typeof storeMsix> | undefined} */
const storeMsixWhenStore = storeMsix

/**
 * The store MSIX packaging identity. Callers must only invoke this when
 * `store` is true (see the invariant note above).
 * @param {NonNullable<typeof storeMsix> | undefined} value
 * @returns {NonNullable<typeof storeMsix>}
 */
function mustStoreMsix(value) {
  return /** @type {NonNullable<typeof storeMsix>} */ (value)
}

// The out-of-store MSIX publisher (ATS cert subject) — single source, shared
// with the .appinstaller generator so the manifest and the App Installer can
// never drift (see scripts/msix-shared.mjs).
const { OUT_OF_STORE_PUBLISHER } = require('../../scripts/msix-shared.mjs')

/** @typedef {import("app-builder-lib").Configuration} Configuration */

const [owner, repo] = (process.env.GITHUB_REPOSITORY || 'NousResearch/hermes-agent').split('/')
if (!owner || !repo) {
  throw new Error(`invalid GITHUB_REPOSITORY ${process.env.GITHUB_REPOSITORY}`)
}
const electronVersion = require('./package.json').devDependencies.electron
if (!/^\d+\.\d+\.\d+$/.test(electronVersion)) {
  throw new Error(`invalid electron version ${electronVersion} in package.json`)
}

/** @type {Configuration} */
module.exports = {
  electronVersion,
  appId,
  productName: displayName,
  executableName: displayName,
  protocols: [
    {
      name: `${displayName} Protocol`,
      schemes: ['hermes']
    }
  ],
  // A store build is archived, never served to a feed — prefix its artifact
  // so it can't collide with the out-of-store MSIX of the same tag/arch, and
  // the release pipeline can keep the two apart.
  artifactName: `${store ? 'Store-' : ''}${appNamePascal}-\${version}-\${os}-\${arch}.\${ext}`,
  icon: 'assets/icon',
  // The electron-updater feed. CI builds set CLOUDFLARE_R2_PUBLIC_URL (the R2
  // public bucket / custom domain) and publish there — the feed yml, blockmaps
  // and installers all live in the same flat R2 bucket, and electron-updater
  // resolves the yml's relative artifact paths against it. Builds without the
  // var (local, or a fork without the R2 vars) keep the github provider, which
  // is exactly today's behavior. The store build has no feed at all (the Store
  // owns its distribution and updates).
  publish: store
    ? null
    : [
        process.env.CLOUDFLARE_R2_PUBLIC_URL
          ? { provider: 'generic', url: process.env.CLOUDFLARE_R2_PUBLIC_URL.replace(/\/+$/, ''), channel }
          : { provider: 'github', owner, repo, channel }
      ],
  extraMetadata: {
    name: appNamePascal,
    desktopName: appId
  },
  directories: {
    output: 'release'
  },
  files: ['dist/**', 'assets/**', 'public/**', 'package.json'],
  beforeBuild: 'scripts/before-build.mjs',
  beforePack: 'scripts/before-pack.mjs',
  afterPack: 'scripts/after-pack.mjs',
  ...(process.platform === 'darwin' ? { afterSign: 'scripts/notarize.mjs' } : {}),
  extraResources: [
    {
      from: 'build/install-stamp.json',
      to: 'install-stamp.json'
    },
    {
      from: 'build/agent-payload',
      to: 'agent-payload'
    },
    {
      from: 'assets/icon.ico',
      to: 'icon.ico'
    }
  ],
  asar: {
    unpack: ['**/*.node', '**/prebuilds/**', 'dist/**']
  },
  mac: {
    category: 'public.app-category.developer-tools',
    extendInfo: {
      CFBundleDisplayName: displayName,
      CFBundleExecutable: displayName,
      CFBundleName: displayName,
      LSRequiresNativeExecution: true,
      NSAudioCaptureUsageDescription: `${displayName} uses audio capture for voice conversations.`,
      NSCameraUsageDescription: `${displayName} uses the camera when a plugin or feature you enable requests it.`,
      NSMicrophoneUsageDescription: `${displayName} uses the microphone for voice input and voice conversations.`,
      NSCalendarsUsageDescription: `${displayName} needs access to Calendar to provide requested meeting and scheduling support.`,
      NSCalendarsFullAccessUsageDescription: `${displayName} needs full access to Calendar to read and manage events when explicitly requested.`,
      NSRemindersUsageDescription: `${displayName} needs access to Reminders to provide requested personal-assistant and scheduling support.`,
      NSRemindersFullAccessUsageDescription: `${displayName} needs full access to Reminders to read and manage reminders when explicitly requested.`,
      NSScreenCaptureUsageDescription: `${displayName} captures the screen when you ask the agent to screenshot or record it.`,
      NSLocalNetworkUsageDescription: `${displayName} connects to devices on your local network when a plugin or feature you enable requests it.`,
      NSAppleMusicUsageDescription: `${displayName} accesses your music library when a plugin or feature you enable requests it.`
    },
    target: ['dmg', 'zip'],
    sign: {
      entitlements: 'electron/entitlements.mac.plist',
      entitlementsInherit: 'electron/entitlements.mac.inherit.plist',
      hardenedRuntime: true,
      ignore: (/** @type {string} */ file) => {
        try {
          if (fs.lstatSync(file).isDirectory()) {
            return false
          }
          return !isMachO(file)
        } catch {
          return true
        }
      }
    }
  },
  dmg: {
    title: `Install ${displayName}`,
    backgroundColor: '#f5f5f7',
    iconSize: 96,
    window: {
      width: 560,
      height: 360
    },
    contents: [
      {
        x: 160,
        y: 170,
        type: 'file'
      },
      {
        x: 400,
        y: 170,
        type: 'link',
        path: '/Applications'
      }
    ]
  },
  win: {
    legalTrademarks: displayName,
    target: ['msix'],
    ...windowsSigning()
  },
  msix: {
    // A store build uses the Partner Center packaging identity (the Store
    // re-signs + rewrites the publisher on submission); everything else uses
    // the out-of-store ATS-cert identity.
    identityName: store ? mustStoreMsix(storeMsixWhenStore).identityName : msixAppIdWithOrg,
    applicationId: appNamePascal,
    displayName,
    publisher: store ? mustStoreMsix(storeMsixWhenStore).publisher : OUT_OF_STORE_PUBLISHER,
    publisherDisplayName: store ? mustStoreMsix(storeMsixWhenStore).publisherDisplayName : 'Nous Research',
    // Canary MSIX versions are `X.Y.Z.<minutes-since-stable>` (see
    // scripts/msix-shared.mjs). setBuildNumber makes getVersionInWeirdWindowsForm
    // use the BUILD_NUMBER env (4th component) instead of hardcoding ".0" — a
    // stable build sets no BUILD_NUMBER and stays X.Y.Z.0, a canary build sets
    // it via build-bundled-desktop.mjs so App Installer updates over equal
    // canary-over-canary versions instead of refusing them.
    setBuildNumber: true,
    // Floor Windows 11 22H2. Below build 18307 the manifest schema caps
    // AppExtension Name at 39 chars and Microsoft's own
    // "com.microsoft.windows.copilotkeyprovider" is 40 (makeappx
    // 0x80080204 — A/B-verified against the 26100 kit; 18307 exactly
    // still failed on it, 22621 passes), and 22621 is the documented
    // Copilot hardware key floor anyway.
    minVersion: '10.0.22621.0',
    maxVersionTested: '10.0.26100.0',
    // Static path: the file itself is written by scripts/before-build.mjs at
    // build time (see the comment on the hook) — never at config require
    // time, so typecheck/test imports don't touch the filesystem.
    customExtensionsPath: 'build/msix-extensions.xml',
    customManifestPath: 'assets/msix-manifest.xml',
    showNameOnTiles: true
  },
  linux: {
    category: 'Development',
    maintainer: 'Nous Research <support@nousresearch.com>',
    synopsis: light
      ? 'Remote-only desktop client for Hermes Agent.'
      : 'Native desktop shell for Hermes Agent.',
    target: ['AppImage']
  }
}

// MSIX build-time staging (build/appx icons + build/msix-extensions.xml)
// lives in scripts/before-build.mjs — an electron-builder lifecycle hook —
// NOT here at require time, so importing this config for typecheck/tests
// never writes to the filesystem.

// Azure Trusted Signing. The hook signs the .msix package itself and the
// product exe (after electron-builder's rcedit — the batch in afterPack runs
// too early for the exe); every other payload binary is batch-signed in
// afterPack via scripts/batch-sign-binaries.mjs, which this hook acknowledges
// with `true` so electron-builder never re-signs one-by-one. The package
// signature is all Windows install validation checks; inner Authenticode
// covers SmartScreen/WDAC tree scans. See batch-sign-binaries.mjs for the
// ordering contract.
const MACHO_MAGICS = new Set([
  0xfeedface,
  0xcefaedfe,
  0xfeedfacf,
  0xcffaedfe,
  0xcafebabe,
  0xbebafeca
])

/** @param {string} file */
function isMachO(file) {
  const buf = Buffer.alloc(4)
  const fd = fs.openSync(file, 'r')
  try {
    if (fs.readSync(fd, buf, 0, 4, 0) !== 4) {
      return false
    }
  } finally {
    fs.closeSync(fd)
  }
  return MACHO_MAGICS.has(buf.readUInt32BE(0))
}

function windowsSigning() {
  if (!process.env.AZURE_SIGN_ENDPOINT || !process.env.AZURE_CLIENT_ID) {
    return {}
  }
  return {
    sign: {
      type: 'signtool',
      sign: './scripts/batch-sign-binaries.mjs',
      signingHashAlgorithms: ['sha256'],
      publisherName: process.env.AZURE_SIGN_PUBLISHER
    }
  }
}

#!/usr/bin/env node
// gen-appinstaller.mjs — generate the Windows App Installer (.appinstaller)
// file for an out-of-store MSIX channel feed.
//
// The out-of-store distribution is App Installer owned: each stable/canary
// channel dir under the feed host holds a universal .msixbundle plus an
// .appinstaller that (a) installs the bundle and (b) records the .appinstaller
// URI as the package's update source, so the OS can re-check it on launch.
//
// The identity comes from product-identity.cjs via scripts/msix-shared.mjs
// (the SAME single derivation as the package manifest), so the
// .appinstaller's MainPackage Name/Publisher always match the bundle's
// manifest. `store` has no appinstaller (the Store owns its distribution).
//
// Pure buildAppInstaller() lives in scripts/msix-shared.mjs and is
// unit-tested; this module is the CLI wrapper:
//   node apps/desktop/scripts/gen-appinstaller.mjs --out <file> --base-url <url>
// Reads HERMES_DESKTOP_VARIANT (bundled|light), HERMES_PAYLOAD_TAG (channel)
// and package.json (version) like the rest of the build.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { appIdentity, buildAppInstaller } from '../../../scripts/msix-shared.mjs'

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const isCli = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href

if (isCli) {
  // node strips the first '--' (and an immediately-following option) for its
  // own use; --out lands as a BARE arg. Parse space-separated flag pairs,
  // not --flag=value, so the CLI survives that mangling.
  const args = process.argv.slice(2)
  const flagValue = (name) => {
    for (let i = 0; i < args.length - 1; i += 1) {
      if (args[i] === name) return args[i + 1]
    }
    return undefined
  }
  const out = flagValue('--out')
  const baseUrl = flagValue('--base-url') || process.env.CLOUDFLARE_R2_PUBLIC_URL

  if (!out || !baseUrl) {
    console.error('[gen-appinstaller] --out=<file> and --base-url=<url> (or CLOUDFLARE_R2_PUBLIC_URL) are required')
    process.exit(1)
  }

  const { identity, version, name } = appIdentity(desktop, process.env.HERMES_PAYLOAD_TAG)
  if (identity.store) {
    console.error('[gen-appinstaller] a store build has no App Installer feed (the Store owns distribution)')
    process.exit(1)
  }

  const canary = /-canary/.test(process.env.HERMES_PAYLOAD_TAG || '')
  const variantDir = identity.light ? 'light/' : ''
  const ch = canary ? 'canary' : 'stable'
  const channelPath = `win32/${variantDir}${ch}`

  const xml = buildAppInstaller({
    baseUrl: String(baseUrl).replace(/\/+$/, ''),
    variantChannelPath: channelPath,
    identityName: identity.msixAppIdWithOrg,
    version,
    bundleFilename: `${name}-${version}-win.msixbundle`
  })

  fs.mkdirSync(path.dirname(out), { recursive: true })
  fs.writeFileSync(out, xml)
  console.log(`[gen-appinstaller] wrote ${out} (${channelPath}/${name}-${version}-win.msixbundle)`)
}

// Generate the AppxManifest.xml that MsixTarget.writeManifest produces on
// the win32 CI lane, using the REAL app-builder-lib helpers and the real
// builder config, so the manifest can be inspected off-Windows — makeappx
// reports every manifest problem as a bare 0x80080204, and this is the
// XML it is rejecting.
//
// Usage (from apps/desktop):
//   node scripts/gen-msix-manifest.mjs [bundled|light] [x64|arm64]
//
// The output is the substituted manifest on stdout. Imports reach into
// app-builder-lib's dist because the exports map hides the internals; the
// paths are the same ones MsixTarget itself uses, so a bump that moves
// them fails here loudly.
import { readFile, readdir } from "node:fs/promises"
import { createRequire } from "node:module"
import path from "node:path"
import { fileURLToPath } from "node:url"

import {
  buildCapabilitiesXml,
  buildExtensionsXml,
  defaultTileTag,
  lockScreenTag,
  resolvePackageApplicationId,
  resolvePackageIdentityName,
  resourceLanguageTag,
  splashScreenTag,
  substituteManifestMacros,
} from "../../../node_modules/app-builder-lib/dist/targets/win/winAppUtil.js"
import { appIdentity } from "../../../scripts/msix-shared.mjs"

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const repoRoot = path.resolve(desktop, "..", "..")
const variant = process.argv[2] || "bundled"
const arch = process.argv[3] || "x64"

if (!["bundled", "light", "store"].includes(variant)) {
  console.error(`variant must be 'bundled', 'light', or 'store', got '${variant}'`)
  process.exit(1)
}

process.env.HERMES_DESKTOP_VARIANT = variant
const require = createRequire(import.meta.url)
const config = require(path.join(desktop, "electron-builder.config.cjs"))
const pkg = require(path.join(desktop, "package.json"))

const options = config.msix
// The payload CLI launcher (hermes.exe, the minted distlib launcher) is the
// MSIX entry point; the alias and AppExtension fragments in the config
// reference the same path. The launcher is self-relative and sets the
// payload's own PYTHONPATH.
const executable = `app\\resources\\agent-payload\\bin\\hermes.exe`
const displayName = options.displayName || config.productName
// appInfo.name honours extraMetadata.name the way packager merging does.
const appInfoName = config.extraMetadata?.name || pkg.name

// Same asset discovery as computeUserAssets: requiring the config staged
// assets/appx/ into build/appx/ (the buildResources "appx" dir MsixTarget
// reads), so listing it here sees exactly what the win32 lane packs.
const userAssets = (await readdir(path.join(desktop, "build", "appx")))
  .filter((it) => !it.startsWith(".") && !it.endsWith(".db") && it.includes("."))

const extensions = await buildExtensionsXml({
  protocols: config.protocols,
  fileAssociations: [],
  addAutoLaunchExtension: options.addAutoLaunchExtension,
  customExtensionsPath: options.customExtensionsPath,
  appDir: desktop,
  executable,
  displayName,
  dependencyNames: {},
})

// Same precedence as MsixTarget.writeManifest: the config's custom
// template when set (resolved relative to the app dir, as getResource
// does), else the stock template.
const template = await readFile(
  options.customManifestPath
    ? path.resolve(desktop, options.customManifestPath)
    : path.join(repoRoot, "node_modules", "app-builder-lib", "templates", "msix", "appxmanifest.xml"),
  "utf8"
)

const manifest = substituteManifestMacros(template, (m) => {
  switch (m) {
    case "publisher": return options.publisher
    case "publisherDisplayName": return options.publisherDisplayName
    // Same 4-part derivation the real build uses (msix-shared::appIdentity) —
    // a canary shows its minutes-since-stable component here too, so this
    // inspection tool never disagrees with what electron-builder shipped.
    case "version": return appIdentity(desktop, process.env.HERMES_PAYLOAD_TAG).version
    case "applicationId": return resolvePackageApplicationId(options.applicationId, options.identityName, appInfoName, "MSIX")
    case "identityName": return resolvePackageIdentityName(options.identityName, appInfoName, "MSIX")
    case "executable": return executable
    case "displayName": return displayName
    case "description": return pkg.description || config.productName
    case "backgroundColor": return options.backgroundColor || "#464646"
    case "logo": return "assets\\StoreLogo.png"
    case "square150x150Logo": return "assets\\Square150x150Logo.png"
    case "square44x44Logo": return "assets\\Square44x44Logo.png"
    case "lockScreen": return lockScreenTag(userAssets)
    case "defaultTile": return defaultTileTag(userAssets, options.showNameOnTiles || false)
    case "splashScreen": return splashScreenTag(userAssets)
    case "arch": return arch
    case "resourceLanguages": return resourceLanguageTag(options.languages)
    case "capabilities": return `<Capabilities>\n${buildCapabilitiesXml(options.capabilities)}\n</Capabilities>`
    case "extensions": return extensions
    // Win10 1903 (build 18362) floor: desktop6:Service — the MSIX-shipped
    // HermesGateway Windows Service — requires 1903+; settled 2026-09-03.
    // 1809 is a 2018 OS; the bump rides the release notes.
    case "minVersion": return options.minVersion || "10.0.18362.0"
    case "maxVersionTested": return options.maxVersionTested || options.minVersion || "10.0.18362.0"
    case "packageIntegrity": return ""
    default: throw new Error(`Macro ${m} is not defined`)
  }
})

console.log(manifest)

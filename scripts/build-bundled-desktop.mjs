#!/usr/bin/env node
// build-bundled-desktop.mjs — build the fully bundled desktop installer
// locally, on any of the three platforms.
//
//   1. preflight: uv, git, npm exist; a release tag is resolvable
//   2. npm ci at the repo root
//   3. build ui-tui (with hermes-ink) and the dashboard SPA
//   4. download the payload node dist (the exact host node version)
//   5. npm run build in apps/desktop with HERMES_DESKTOP_BUNDLED=1
//   6. npm run builder -- <platform targets>
//
// Every step always runs. There is no opt-out: a skipped step is a
// different artifact, and a different artifact is not a reproduction.
//
// Usage:
//   node scripts/build-bundled-desktop.mjs --tag=v0.20.0
//
// Signing is CI's job (Azure/Apple secrets). Local builds are unsigned.

import { execSync, spawnSync } from "node:child_process"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { hostTarBin } from "../apps/desktop/scripts/stage-agent-payloads.mjs"

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")

const args = process.argv.slice(2)
const tagArg = args.find((a) => a.startsWith("--tag="))?.slice("--tag=".length)
// Everything after `--` goes to electron-builder verbatim (CI appends its
// signing configuration this way).
const dashDash = process.argv.indexOf("--")
const extraBuilderArgs = dashDash === -1 ? [] : process.argv.slice(dashDash + 1)

for (const retired of ["--no-install", "--no-package"]) {
  if (args.includes(retired)) {
    fail(`${retired} is retired: the build always runs every step`)
  }
}

function fail(message) {
  console.error(`[build-bundled] ${message}`)
  process.exit(1)
}

function run(cmd, argv, opts = {}) {
  console.log(`[build-bundled] $ ${cmd} ${argv.join(" ")}`)
  // shell mode is for npm.cmd on Windows. It forbids arguments with
  // spaces: cmd.exe re-splits them and no quoting survives npm's own
  // re-spawn. Anything space-valued must travel as an environment
  // variable instead (see run-electron-builder.mjs for signing).
  const shell = process.platform === "win32"
  if (shell) {
    const bad = argv.find((a) => /\s/.test(a))
    if (bad) {
      fail(`argument with whitespace cannot cross the Windows shell: ${JSON.stringify(bad)} — pass it via environment instead`)
    }
  }
  const result = spawnSync(cmd, argv, { stdio: "inherit", cwd: REPO_ROOT, shell, ...opts })
  if (result.status !== 0) {
    fail(`${cmd} exited ${result.status}`)
  }
}

function capture(cmd) {
  return execSync(cmd, { cwd: REPO_ROOT, encoding: "utf8" }).trim()
}

// ── 1. preflight ────────────────────────────────────────────────────────────

for (const tool of ["uv", "git", "npm", "tar"]) {
  const probe = spawnSync(tool, ["--version"], { stdio: "ignore", shell: process.platform === "win32" })
  if (probe.status !== 0) {
    fail(`required tool missing: ${tool}`)
  }
}

// Toolchain gates. The build's output depends on these tools, so a wrong
// version makes a silently different artifact (the first Windows build
// shipped a wrong-arch uv exactly this way). The rules come from ONE
// source — package.json "engines" — and the embedded runtimes are pinned
// to the EXACT host versions the gates approved:
//   node — the payload node dist is downloaded at the host node version.
//   uv   — the staged uv IS the host binary, copied (stageUvAndPython).
//   npm  — ships inside the node dist; it cannot be chosen separately,
//          so the host npm is gated by engines and the payload npm is
//          whatever the pinned node dist bundles.
export function parseVersion(text) {
  const match = String(text).match(/(\d+)\.(\d+)\.(\d+)/)
  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : null
}

export function compareVersions(a, b) {
  for (let i = 0; i < 3; i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i]
  }
  return 0
}

// The subset of semver ranges that package.json engines actually uses:
// space-separated comparators AND together, `||` separates alternatives.
// An unparseable comparator fails closed.
export function satisfiesRange(version, range) {
  return String(range).split("||").some((alternative) => {
    const comparators = alternative.trim().split(/\s+/).filter(Boolean)
    if (comparators.length === 0) return false
    return comparators.every((comparator) => {
      const m = comparator.match(/^(>=|<=|>|<|=)?v?(\d+)\.(\d+)\.(\d+)$/)
      if (!m) return false
      const cmp = compareVersions(version, [Number(m[2]), Number(m[3]), Number(m[4])])
      switch (m[1]) {
        case ">=": return cmp >= 0
        case "<=": return cmp <= 0
        case ">": return cmp > 0
        case "<": return cmp < 0
        default: return cmp === 0
      }
    })
  })
}

export function uvBannerProblem(banner) {
  // A build triple is three dash-joined words that end in letters
  // (aarch64-pc-windows-msvc). Its position varies: nix builds print it
  // first in the parens, official builds put a commit hash and a date
  // before it. Match it anywhere — the date (2026-07-31) cannot match
  // because its last segment is digits.
  return /[a-z0-9_]+-[a-z0-9]+-[a-z][a-z0-9-]*/.test(String(banner))
    ? null
    : "its --version prints no build triple; the payload arch guard needs one (official uv 0.12+, or any nix/source build)"
}

const engines = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "package.json"), "utf8")).engines || {}

// The approved host toolchain. Filled by the gates below; the payload
// stages embed THESE versions, so gate == embed by construction.
const HOST_TOOLCHAIN = { node: null, npm: null, uvBanner: null }

for (const tool of ["node", "npm"]) {
  const text = tool === "node" ? process.version : capture("npm --version")
  const version = parseVersion(text)
  if (!version) {
    fail(`${tool}: cannot parse a version from ${JSON.stringify(text)}`)
  }
  const range = engines[tool]
  if (range && !satisfiesRange(version, range)) {
    fail(`${tool} ${version.join(".")} does not satisfy package.json engines ${JSON.stringify(range)} — the build would make a different artifact`)
  }
  HOST_TOOLCHAIN[tool] = version
  console.log(`[build-bundled] ${tool} ${version.join(".")} (engines: ${range || "unconstrained"})`)
}

{
  const uvBanner = capture("uv --version")
  const problem = uvBannerProblem(uvBanner)
  if (problem) {
    fail(`uv (${uvBanner}) would make a broken artifact: ${problem}`)
  }
  HOST_TOOLCHAIN.uvBanner = uvBanner
  console.log(`[build-bundled] ${uvBanner} (staged into the payload as-is)`)
}

let tag = tagArg
if (!tag) {
  try {
    tag = capture("git describe --tags --exact-match")
  } catch {
    fail("no --tag=vX.Y.Z given and HEAD is not at an exact release tag")
  }
}
if (!/^v(?:0|[1-9]\d{0,2})\.\d+\.\d+$/.test(tag)) {
  fail(`'${tag}' is not a final release tag (vX.Y.Z)`)
}

// The canonical Hermes version is owned by pyproject.toml (the same rule
// the Nix derivation applies). electron-builder gets it via extraMetadata,
// so app.getVersion(), the artifact names, and the latest*.yml feed all
// carry the real release version instead of the UI manifest's stale one.
// The tag must agree with it: a v0.21.0 payload inside an app that
// announces 0.20.0 would make electron-updater blind to the mismatch.
const pyprojectVersion = fs
  .readFileSync(path.join(REPO_ROOT, "pyproject.toml"), "utf8")
  .match(/^version\s*=\s*"([^"]+)"/m)?.[1]
if (!pyprojectVersion) {
  fail("could not read version from pyproject.toml")
}
if (tag !== `v${pyprojectVersion}`) {
  fail(`tag ${tag} does not match pyproject.toml version ${pyprojectVersion}`)
}

const targets = { linux: "--linux AppImage", darwin: "--mac dmg zip", win32: "--win nsis" }[process.platform]
if (!targets) {
  fail(`unsupported platform: ${process.platform}`)
}

console.log(`[build-bundled] tag=${tag} platform=${process.platform}-${process.arch}`)

// ── 2-3. deps + JS surfaces ─────────────────────────────────────────────────

// ui-tui, ui-tui/packages/*, and web are npm workspaces of the repo root:
// ONE root `npm ci` installs all of them, hoisted into the root
// node_modules. Never run npm ci inside a workspace directory — that
// builds a partial shadow tree beside the hoisted one and breaks module
// resolution for the workspace builds below.
run("npm", ["ci", "--no-audit", "--no-fund"], {
  env: {
    ...process.env, // spawnSync env REPLACES the child environment; keep PATH etc.
    "CI": "true" // skip annoying unicode install banner
  }
})
run("npm", ["run", "build", "--workspace", "ui-tui"])
run("npm", ["run", "build", "--workspace", "web"])

// ── 4. payload node dist ────────────────────────────────────────────────────
// Pinned to the EXACT host node version: the JS surfaces were built and
// npm-installed by the host node, and the payload node runs them at
// runtime. A different version is a different artifact. This also means
// the host node must be an official nodejs.org release — a patched build
// whose version does not exist upstream fails here, loudly.

const distName = { linux: "linux", darwin: "darwin", win32: "win" }[process.platform]
const distArch = { x64: "x64", arm64: "arm64" }[process.arch]
const distExt = process.platform === "win32" ? "zip" : process.platform === "darwin" ? "tar.gz" : "tar.xz"

const version = `v${HOST_TOOLCHAIN.node.join(".")}`
const index = JSON.parse(
  execSync(`curl -fsSL https://nodejs.org/dist/index.json`, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 })
)
if (!index.some((e) => e.version === version)) {
  fail(`host node ${version} is not an official nodejs.org release — cannot embed the exact build toolchain`)
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), "hermes-node-payload-"))
const archive = `node-${version}-${distName}-${distArch}.${distExt}`
const extractDir = path.join(work, "extract")
const nodeDir = path.join(work, "node-payload")
fs.mkdirSync(extractDir, { recursive: true })

console.log(`[build-bundled] payload node: ${version}`)
run("curl", ["-fsSL", "-o", path.join(work, archive), `https://nodejs.org/dist/${version}/${archive}`])
run(hostTarBin(), ["-xf", path.join(work, archive), "-C", extractDir])
const [topDir] = fs.readdirSync(extractDir)
fs.renameSync(path.join(extractDir, topDir), nodeDir)

const nodeBinary = process.platform === "win32" ? path.join(nodeDir, "node.exe") : path.join(nodeDir, "bin", "node")
if (!fs.existsSync(nodeBinary)) {
  fail(`extracted node dist has no runnable node at ${nodeBinary}`)
}

// ── 5-6. bundled desktop build + package ────────────────────────────────────

const env = {
  ...process.env,
  HERMES_DESKTOP_BUNDLED: "1",
  HERMES_PAYLOAD_TAG: tag,
  HERMES_PAYLOAD_PYTHON: process.env.HERMES_PAYLOAD_PYTHON || "3.11",
  HERMES_PAYLOAD_NODE_DIST: nodeDir,
}

const desktop = path.join(REPO_ROOT, "apps", "desktop")
run("npm", ["run", "build"], { cwd: desktop, env })

run(
  "npm",
  [
    "run", "builder", "--",
    ...targets.split(" "),
    `-c.extraMetadata.version=${pyprojectVersion}`,
    ...extraBuilderArgs,
  ],
  { cwd: desktop, env }
)
console.log(`[build-bundled] artifacts: ${path.join(desktop, "release")}`)

fs.rmSync(work, { recursive: true, force: true })

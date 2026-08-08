/**
 * stage-agent-payloads.mjs: assemble the resources-resident agent runtime
 * that ships inside the bundled desktop artifact. Design:
 * .hermes/plans/2026-08-07_resources-resident-bundled-runtime.md.
 *
 * Output: apps/desktop/build/agent-payload/
 *   manifest.json          schemaVersion, tag, commit, platform, arch
 *   repo/                  plain source tree at the release tag (no .git),
 *                          plus the PREBUILT JS surfaces (ui-tui dist +
 *                          node_modules, web_dist) and the build stamp
 *   uv/                    static uv binary for this platform/arch
 *   python/                uv-managed CPython (python-build-standalone).
 *                          Its own site-packages carries hermes-bundle.pth
 *                          with RELATIVE paths to repo/ and site-packages/,
 *                          so the interpreter resolves the runtime wherever
 *                          the app bundle sits — no venv, no PYTHONPATH.
 *   site-packages/         the full dependency tree from uv.lock, installed
 *                          at build time with `pip install --target` on the
 *                          payload interpreter. The backend runs directly
 *                          from here; nothing materializes at first launch.
 *   node/                  official node dist for this platform/arch
 *
 * Gating: the script does nothing unless HERMES_DESKTOP_BUNDLED=1. That
 * variable is an internal build-time env for CI wiring, not user config.
 * Thus dev builds and current CI keep producing external builds. There is
 * no per-item skip: an embedded payload is complete, or this script throws
 * and the build fails.
 *
 * The heavy work shells out to git, uv, and tar. The decision logic
 * (target resolution, pip arg construction, manifest shape) is exported as
 * pure functions. Thus vitest covers it without network or toolchains.
 */

import { execSync, spawnSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

import { isMain } from "./utils.mjs"

export const PAYLOAD_SCHEMA_VERSION = 3

const DESKTOP_ROOT = path.resolve(import.meta.dirname, "..")
const REPO_ROOT = path.resolve(DESKTOP_ROOT, "..", "..")
const OUT_DIR = path.join(DESKTOP_ROOT, "build", "agent-payload")

/**
 * Map (process.platform, process.arch) to the uv, python-build-standalone,
 * and node target descriptors. There is one artifact per (os, arch) pair.
 * Mac universal2 is deliberately NOT a target. We ship two artifacts
 * (plan §6).
 *
 * There are no cross-platform wheel tags here, on purpose. A CI runner per
 * (os, arch) pair assembles the payloads. electron-builder needs per-OS
 * runners for signing anyway. Thus the script fetches wheels NATIVELY with
 * `uvx pip wheel --only-binary=:all:`. The platform of the runner is the
 * target platform.
 */
export function resolveTargets(platform = process.platform, arch = process.arch) {
  const table = {
    "linux-x64": {
      uvTarget: "x86_64-unknown-linux-gnu",
      pythonPlatform: "x86_64-unknown-linux-gnu",
      nodeDist: "linux-x64",
      uvPython: "linux-x86_64-gnu",
    },
    "linux-arm64": {
      uvTarget: "aarch64-unknown-linux-gnu",
      pythonPlatform: "aarch64-unknown-linux-gnu",
      nodeDist: "linux-arm64",
      uvPython: "linux-aarch64-gnu",
    },
    "darwin-x64": {
      uvTarget: "x86_64-apple-darwin",
      pythonPlatform: "x86_64-apple-darwin",
      nodeDist: "darwin-x64",
      uvPython: "macos-x86_64-none",
    },
    "darwin-arm64": {
      uvTarget: "aarch64-apple-darwin",
      pythonPlatform: "aarch64-apple-darwin",
      nodeDist: "darwin-arm64",
      uvPython: "macos-aarch64-none",
    },
    "win32-x64": {
      uvTarget: "x86_64-pc-windows-msvc",
      pythonPlatform: "x86_64-pc-windows-msvc",
      nodeDist: "win-x64",
      uvPython: "windows-x86_64-none",
    },
    "win32-arm64": {
      uvTarget: "aarch64-pc-windows-msvc",
      pythonPlatform: "aarch64-pc-windows-msvc",
      nodeDist: "win-arm64",
      uvPython: "windows-aarch64-none",
      // Pinned packages with no published win_arm64 wheel. pip builds
      // these from sdist on the runner (needs MSVC arm64 + Rust).
      // pyyaml publishes win_arm64 wheels for cp312+ only — the payload
      // python is 3.11, so it builds here too (pure fallback when the
      // libyaml accelerator is unavailable).
      sourceBuild: ["cryptography", "httptools", "ruamel-yaml-clib", "pywinpty", "pyyaml"],
    },
  }
  const key = `${platform}-${arch}`
  const target = table[key]
  if (!target) {
    throw new Error(`unsupported payload target: ${key}`)
  }
  return { key, platform, arch, ...target }
}

/**
 * Build the `pip install --target` argument list that fills the payload's
 * site-packages. The caller invokes it through `uvx pip …` ON the staged
 * payload interpreter, natively on the target runner, so wheels resolve
 * for the target platform/arch. With --only-binary=:all: it never
 * compiles on the user machine — there IS no install step on the user
 * machine; the backend imports straight from this directory.
 *
 * Exception: the target's sourceBuild list. Some pinned packages publish
 * no wheel for a target (win32-arm64: cryptography dropped win_arm64
 * after 46.0.3; httptools and ruamel-yaml-clib never shipped one;
 * pywinpty 2.x has none). For those named packages pip builds the
 * EXACT pinned version from its sdist ON the build runner, which yields
 * real target-arch code in site-packages — the user machine still
 * never compiles. The build runner needs the toolchains (MSVC arm64 +
 * Rust on windows-11-arm). A later --no-binary overrides --only-binary
 * per package; the list stays empty for every target whose pins are
 * fully covered by published wheels.
 */
export function pipTargetArgs({ sitePackagesDir, sourceBuild = [] }) {
  return [
    "install",
    "--only-binary", ":all:",
    ...(sourceBuild.length > 0 ? ["--no-binary", sourceBuild.join(",")] : []),
    "-r", "requirements-payload.txt",
    "--target", sitePackagesDir,
    // pip warns without this when --target sees an existing dir; staging
    // wipes first, so upgrade semantics never actually apply.
    "--upgrade",
    // No console-script shims: the bundle always launches `python -m`,
    // and --target's scripts would carry the BUILD host's shebang paths.
    "--no-compile",
  ]
}

/**
 * The full uv python-install request for a target: version AND platform.
 * A bare version request ("3.11") lets uv fall back to another
 * architecture when the native build is unavailable — the arm64 Windows
 * test box got a silent x86_64 CPython that way. The full request either
 * installs the right build or fails loudly.
 */
export function pythonRequest(target, version = process.env.HERMES_PAYLOAD_PYTHON || "3.11") {
  return `cpython-${version}-${target.uvPython}`
}

/**
 * Assert that a staged tool's own version banner names the target triple.
 * `uv --version` and `python -VV` both print their build triple/platform.
 * A mismatch means the payload carries the WRONG architecture (for
 * example, an x64 uv copied from PATH into an arm64 artifact — it runs
 * on the build host through emulation and ships broken). The manifest
 * would then lie about the payload's contents. Fail the build instead.
 */
export function assertBanner(item, banner, mustContain) {
  if (!banner.includes(mustContain)) {
    throw new Error(
      `${item}: staged binary reports "${banner.trim()}" which does not ` +
        `contain the build target "${mustContain}" — wrong-architecture ` +
        `payload. Provide a matching binary (HERMES_PAYLOAD_UV for uv) or ` +
        `build on a native runner.`
    )
  }
}

/**
 * The substring that each staged tool's banner must contain for a target.
 * uv prints a full triple (x86_64-pc-windows-msvc). CPython's `python -VV`
 * prints a compiler/platform line that differs per OS, so the check keys
 * on the architecture words for it. Node prints nothing useful in
 * --version, so its check uses `node -p process.arch` = target arch.
 */
export function bannerExpectations(target) {
  const archWords = {
    x64: ["x86_64", "AMD64", "x64"],
    arm64: ["aarch64", "ARM64", "arm64"],
  }[target.arch]

  return {
    uv: target.uvTarget,
    pythonAny: archWords,
    node: target.arch,
  }
}


/**
 * Resolve the release tag to stage. CI passes --tag=vX.Y.Z. Local runs can
 * fall back to `git describe` for smoke tests. When bundling was requested
 * and no tag exists, payload staging is a hard error. A bundled artifact
 * without a pinned tag produces un-updatable installs.
 */
export function resolveTag(argv, describeFn) {
  const explicit = argv.find((a) => a.startsWith("--tag="))
  if (explicit) {
    const tag = explicit.slice("--tag=".length).trim()
    if (!/^v(?:0|[1-9]\d{0,2})\.\d+\.\d+$/.test(tag)) {
      throw new Error(`--tag must be a final release tag (vX.Y.Z), got: ${tag}`)
    }
    return tag
  }
  const described = describeFn()
  if (described && /^v(?:0|[1-9]\d{0,2})\.\d+\.\d+$/.test(described)) {
    return described
  }
  throw new Error(
    "no release tag: pass --tag=vX.Y.Z (CI) or run from a checkout at an exact release tag"
  )
}

/**
 * Build the manifest that marks a complete embedded payload. The Electron
 * main process treats its presence (schemaVersion match, external: absent)
 * as the payload-present sentinel. Completeness is a build-time invariant:
 * main() throws before this manifest is written when any stage fails.
 */
export function buildManifest({ tag, commit, target }) {
  return {
    schemaVersion: PAYLOAD_SCHEMA_VERSION,
    tag,
    commit,
    platform: target.platform,
    arch: target.arch,
    builtAt: new Date().toISOString(),
  }
}

// ─── impure staging steps (they shell out, have no unit tests, and run in CI) ──────

function run(cmd, args, opts = {}) {
  // stdio: inherit — subprocess output (pip's resolution errors, uv's
  // install messages) streams to the build log in real time. The throw
  // below only names the command; the CAUSE is in the streamed output
  // directly above it.
  const result = spawnSync(cmd, args, { stdio: "inherit", ...opts })
  if (result.error) {
    throw new Error(`${cmd} did not start: ${result.error.message}`)
  }
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} exited ${result.status} — its error output is printed above`)
  }
}

/**
 * Capture a probe command's stdout for inspection (banner checks). On
 * failure the captured stderr goes into the thrown error, so probe
 * failures are never silent.
 */
function probe(cmd, args) {
  const result = spawnSync(cmd, args, { encoding: "utf8" })
  if (result.error) {
    throw new Error(`${cmd} did not start: ${result.error.message}`)
  }
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} exited ${result.status}: ${(result.stderr || "").trim()}`)
  }
  return result.stdout
}

function stageRepo(tag, outDir) {
  const repoDir = path.join(outDir, "repo")
  fs.rmSync(repoDir, { recursive: true, force: true })
  fs.mkdirSync(repoDir, { recursive: true })
  // rev-list, not `rev-parse <tag>^{commit}`: execSync on Windows runs
  // through cmd.exe, where ^ is the escape character and eats the brace.
  const commit = execSync(`git rev-list -n 1 ${tag}`, { cwd: REPO_ROOT, encoding: "utf8" }).trim()
  const commitDate = execSync(`git log -1 --format=%ct ${tag}`, { cwd: REPO_ROOT, encoding: "utf8" }).trim()
  // The payload repo is a PLAIN SOURCE TREE, deliberately without .git.
  // Bundled installs never run git against the checkout: updates replace
  // the whole tree (electron-updater), and `hermes update --eject` makes
  // its own fresh clone. A shipped .git also broke in transit: `git gc`
  // packs all refs, which leaves .git/refs/ empty, and electron-builder's
  // resource copy drops empty directories — git then refuses to recognize
  // the repository at all. git archive gives a clean tree of exactly the
  // tag's tracked files.
  const archive = path.join(outDir, ".repo-archive.tar")
  run("git", ["archive", "--format=tar", "-o", archive, tag], { cwd: REPO_ROOT })
  run(hostTarBin(), ["-xf", archive, "-C", repoDir])
  fs.rmSync(archive, { force: true })
  // The PREBUILT JS surfaces live inside the repo tree, exactly where a
  // source checkout builds them. CI builds ui-tui (with hermes-ink) and
  // the dashboard SPA BEFORE this script runs; here they are copied in
  // as plain directories. The SPA's real outDir is hermes_cli/web_dist
  // (web/vite.config.ts) — the old js-prebuilt list named a root-level
  // web_dist that never existed, and its existsSync filter silently
  // dropped it from every artifact. dereference: ui-tui/node_modules
  // carries the hermes-ink workspace symlink, and symlinks do not
  // reliably survive the electron-builder resource copy.
  const jsSurfaces = ["ui-tui/dist", "ui-tui/node_modules", "hermes_cli/web_dist"].filter((p) =>
    fs.existsSync(path.join(REPO_ROOT, p))
  )
  if (jsSurfaces.length < 3) {
    throw new Error(`repo: prebuilt JS surfaces missing — run the ui-tui/web builds first (found: ${jsSurfaces.join(", ") || "none"})`)
  }
  for (const surface of jsSurfaces) {
    fs.cpSync(path.join(REPO_ROOT, surface), path.join(repoDir, surface), {
      recursive: true,
      dereference: true,
    })
  }
  // Version provenance without git: the schema-v2 build stamp. The
  // version_info ladder prefers this stamp over git probing, so bundled
  // installs report exact-release provenance (distance 0, the tag's
  // commit) with no .git present.
  // uv run, not bare python3: on Windows `python3` resolves to the
  // Microsoft Store alias (exit 9009). uv is a hard prerequisite of this
  // script anyway, and the desktop `build` npm script already runs this
  // same stamp writer through it.
  run("uv", [
    "run", "--no-project", "--python", "3",
    path.join(repoDir, "scripts", "write_install_stamp.py"),
    "--output", path.join(repoDir, ".hermes_build_info.json"),
    "--commit", commit,
    "--commit-date", commitDate,
    "--base-version", tag.slice(1),
    "--distance", "0",
    "--source", "ci",
    "--distribution", "desktop-app",
  ])
  return commit
}

// Windows: name System32's bsdtar by full path. A GNU tar earlier on
// PATH (Git bash on the GitHub runners) reads "C:" in a path as a
// remote host name. bsdtar also reads .zip, so one extraction call
// covers every archive format the payload pipeline downloads.
export function hostTarBin() {
  return process.platform === "win32"
    ? path.join(process.env.SystemRoot || "C:\\Windows", "System32", "tar.exe")
    : "tar"
}

function stageUvAndPython(target, outDir) {
  const uvDir = path.join(outDir, "uv")
  const pythonDir = path.join(outDir, "python")
  // Wipe before staging (stageRepo does the same). A rerun after a failed
  // or wrong-arch attempt must not leave a stale interpreter beside the
  // new one — the banner probe would find the old build first.
  fs.rmSync(uvDir, { recursive: true, force: true })
  fs.rmSync(pythonDir, { recursive: true, force: true })
  fs.mkdirSync(uvDir, { recursive: true })
  fs.mkdirSync(pythonDir, { recursive: true })
  // Native runner: the uv that runs this build IS the target-platform uv.
  // HERMES_PAYLOAD_UV overrides this for unusual setups. The default is
  // `uv` on PATH.
  const uvName = target.platform === "win32" ? "uv.exe" : "uv"
  const uvSource =
    process.env.HERMES_PAYLOAD_UV ||
    execSync(
      target.platform === "win32" ? "where uv" : "command -v uv",
      { encoding: "utf8" }
    ).split(/\r?\n/)[0].trim()
  const uvStaged = path.join(uvDir, uvName)
  fs.copyFileSync(uvSource, uvStaged)

  const expect = bannerExpectations(target)

  // The staged uv must be built FOR the target triple, not merely run on
  // this host (emulation makes a wrong-arch binary run fine here).
  // uv prints its build triple in --version from 0.12 on; an older uv
  // prints only the version number, which is unverifiable — refuse it
  // with a message that says so instead of claiming a wrong arch.
  const uvBanner = probe(uvStaged, ["--version"])
  if (/^uv \d[\d.]*\s*$/.test(uvBanner.trim())) {
    throw new Error(
      `uv: "${uvBanner.trim()}" prints no build triple, so its architecture ` +
        `cannot be verified. Use uv 0.12 or newer.`
    )
  }
  assertBanner("uv", uvBanner, expect.uv)

  // --no-bin: staging must not write launcher shims into the build
  // host's ~/.local/bin (it collided with a preexisting python3.11.exe
  // on the Windows test box).
  run("uv", ["python", "install", "--no-bin", "--install-dir", pythonDir, pythonRequest(target)])

  // The installed CPython proves its architecture at runtime.
  // `python -VV` names the arch on Windows ("[MSC v.1944 64 bit (ARM64)]")
  // but not on Linux/macOS ("[Clang 22.1.3 ]"), so the check asks
  // platform.machine() — the value the binary itself reports. The
  // install-directory pattern above already pins the requested build;
  // this is the runtime backstop.
  const pythonBinary = findPythonBinary(pythonDir, target)
  const pythonMachine = probe(pythonBinary, ["-c", "import platform; print(platform.machine())"])
  if (!expect.pythonAny.some((word) => pythonMachine.includes(word))) {
    assertBanner("python", pythonMachine, expect.pythonAny.join("|"))
  }
  return pythonBinary
}

/**
 * Match the directory `uv python install` creates for a request. The
 * request names a minor version (cpython-3.11-windows-aarch64-none), and
 * uv installs into a PATCH-versioned directory
 * (cpython-3.11.15-windows-aarch64-none) plus a minor-version alias that
 * is a junction on Windows. The matcher accepts both shapes and nothing
 * of any other version or triple.
 */
export function pythonDirPattern(target, version = process.env.HERMES_PAYLOAD_PYTHON || "3.11") {
  const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  return new RegExp(`^cpython-${escape(version)}(\\.\\d+)?(rc\\d+)?-${escape(target.uvPython)}$`)
}

function findPythonBinary(pythonDir, target) {
  // Search only directories that match the REQUESTED build, so a stray
  // install of another architecture can never satisfy the probe. The
  // wipe above prevents strays; this is the backstop. The alias
  // entry is a junction/symlink — do not require isDirectory().
  const name = target.platform === "win32" ? "python.exe" : "python3"
  const pattern = pythonDirPattern(target)
  const roots = fs
    .readdirSync(pythonDir, { withFileTypes: true })
    .filter((e) => (e.isDirectory() || e.isSymbolicLink()) && pattern.test(e.name))
    .map((e) => path.join(pythonDir, e.name))
  if (roots.length === 0) {
    throw new Error(`python: nothing matching ${pattern} under ${pythonDir} after uv python install`)
  }
  const stack = [...roots]
  while (stack.length) {
    const dir = stack.pop()
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        stack.push(full)
      } else if (entry.name === name) {
        return full
      }
    }
  }
  throw new Error(`python: no ${name} found under ${roots.join(", ")}`)
}

function stageSitePackages(target, outDir, pythonBinary) {
  const sitePackagesDir = path.join(outDir, "site-packages")
  fs.rmSync(sitePackagesDir, { recursive: true, force: true })
  fs.mkdirSync(sitePackagesDir, { recursive: true })
  // Export the lock to a requirements file, then install the whole tree
  // with pip running ON THE STAGED PAYLOAD INTERPRETER: pip resolves
  // platform tags for the interpreter that executes it, so this is what
  // pins site-packages to the target architecture. (uvx pip runs under
  // uvx's own python — on the arm64 test box that pulled win_amd64
  // wheels.) No venv anywhere: a venv's bin/python is a symlink to an
  // ABSOLUTE build-host path, and the .app runs from unpredictable
  // locations (renames, Gatekeeper translocation, AppImage mounts).
  if (!pythonBinary) {
    throw new Error("site-packages: the uv/python stage must run first (it provides the payload interpreter)")
  }
  run("uv", ["export", "--frozen", "--no-emit-project", "-o", "requirements-payload.txt"], { cwd: REPO_ROOT })
  run(
    "uvx",
    ["--python", pythonBinary, "pip", ...pipTargetArgs({ sitePackagesDir, sourceBuild: target.sourceBuild || [] })],
    { cwd: REPO_ROOT }
  )

  // hermes-agent's own code imports from repo/ (the .pth puts it first on
  // sys.path — PROJECT_ROOT derivations need the real tree around the
  // packages). But importlib.metadata.version("hermes-agent") needs a
  // dist-info. pip cannot produce one here: setup.py deliberately blocks
  // wheel builds outside Nix (and pip install --target builds a wheel
  // internally). importlib.metadata only reads METADATA, so write the
  // minimal dist-info directly — same trick as flat layouts everywhere.
  const version = probe(pythonBinary, [
    "-c",
    `import pathlib, re; print(re.search(r'__version__ = \"([^\"]+)\"', pathlib.Path(${JSON.stringify(
      path.join(outDir, "repo", "hermes_cli", "__init__.py")
    )}).read_text(encoding="utf-8")).group(1))`,
  ]).trim()
  const distInfo = path.join(sitePackagesDir, `hermes_agent-${version}.dist-info`)
  fs.mkdirSync(distInfo, { recursive: true })
  fs.writeFileSync(
    path.join(distInfo, "METADATA"),
    `Metadata-Version: 2.1\nName: hermes-agent\nVersion: ${version}\n`
  )
  fs.writeFileSync(path.join(distInfo, "INSTALLER"), "hermes-desktop-bundle\n")

  // Architecture backstop: import the heaviest native extensions with
  // site-packages on the path. On the native CI runner a wrong-arch
  // tree fails here instead of on the user machine. (The old wheelhouse
  // filename check has no equivalent — pip already unpacked the wheels —
  // and actually importing is the stronger proof.)
  probe(pythonBinary, [
    "-c",
    `import sys; sys.path.insert(0, ${JSON.stringify(sitePackagesDir)}); import pydantic_core, cryptography, charset_normalizer`,
  ])
}

/**
 * The relative sys.path entries for the bundle glue. A .pth file's
 * non-import lines are resolved against the DIRECTORY CONTAINING THE
 * .PTH FILE, so relative entries make the payload fully relocatable:
 * no absolute paths exist anywhere in the artifact. repo/ comes first
 * so its packages win over anything in site-packages.
 */
export function bundlePthLines(purelibDir, payloadRoot, pathModule = path) {
  return ["repo", "site-packages"].map((entry) =>
    pathModule.relative(purelibDir, pathModule.join(payloadRoot, entry))
  )
}

function writeBundlePth(outDir, pythonBinary) {
  // Ask the interpreter where its own site-packages lives instead of
  // hardcoding the layout (POSIX: lib/python3.11/site-packages,
  // Windows: Lib/site-packages).
  const purelib = probe(pythonBinary, ["-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]).trim()
  if (!purelib || !fs.existsSync(purelib)) {
    throw new Error(`bundle pth: interpreter reports nonexistent purelib: ${purelib}`)
  }
  fs.writeFileSync(
    path.join(purelib, "hermes-bundle.pth"),
    bundlePthLines(purelib, outDir).join("\n") + "\n"
  )
}

function stageNode(target, outDir) {
  const nodeDir = path.join(outDir, "node")
  // Idempotent: a leftover tree from an interrupted run makes cpSync
  // throw EEXIST on directory merges; start clean every time.
  fs.rmSync(nodeDir, { recursive: true, force: true })
  fs.mkdirSync(nodeDir, { recursive: true })
  const src = process.env.HERMES_PAYLOAD_NODE_DIST
  if (!src) {
    throw new Error("HERMES_PAYLOAD_NODE_DIST must point at the extracted node dist for the target")
  }
  fs.cpSync(src, nodeDir, { recursive: true })

  // The dist must be FOR the target. Running the staged node is not a
  // valid probe here: a wrong-arch binary can still run through the
  // build host's emulation. `node -p process.arch` names the arch the
  // binary was BUILT for, so execute it only to read that value; when
  // the binary cannot run at all, that is the same wrong-arch verdict.
  const nodeBinary = target.platform === "win32" ? path.join(nodeDir, "node.exe") : path.join(nodeDir, "bin", "node")
  let reportedArch = null
  try {
    reportedArch = probe(nodeBinary, ["-p", "process.arch"]).trim()
  } catch {
    // Unrunnable on this host — for example an arm64 dist on an x64
    // builder with no emulation. That is not proof of a wrong payload,
    // but it IS unverifiable; refuse rather than ship unchecked.
    throw new Error(`node: staged binary at ${nodeBinary} did not run, so its architecture is unverified`)
  }
  assertBanner("node", reportedArch, bannerExpectations(target).node)
}

function main() {
  if (process.env.HERMES_DESKTOP_BUNDLED !== "1") {
    // Thin build: write a stub manifest anyway. Then the extraResources
    // entry always has a real directory to copy. The behavior of
    // electron-builder for a missing `from` changes between versions. The
    // stub also lets runtime code read manifest.json uniformly and learn
    // that there are no payloads.
    fs.mkdirSync(OUT_DIR, { recursive: true })
    fs.writeFileSync(
      path.join(OUT_DIR, "manifest.json"),
      JSON.stringify({ schemaVersion: PAYLOAD_SCHEMA_VERSION, external: true }, null, 2) + "\n"
    )
    console.log("[stage-agent-payloads] HERMES_DESKTOP_BUNDLED != 1 — wrote external stub manifest")
    return
  }
  const target = resolveTargets()
  const tag = resolveTag(process.argv.slice(2), () => {
    try {
      return execSync("git describe --tags --exact-match", { cwd: REPO_ROOT, encoding: "utf8" }).trim()
    } catch {
      return null
    }
  })

  fs.mkdirSync(OUT_DIR, { recursive: true })

  // Every stage runs, in order. A failure throws and the build fails:
  // an embedded payload is complete, or it does not exist.
  console.log(`[stage-agent-payloads] staging: repo (${target.key}, ${tag})`)
  const commit = stageRepo(tag, OUT_DIR)
  console.log(`[stage-agent-payloads] staging: uv + python (${target.key}, ${tag})`)
  const payloadPython = stageUvAndPython(target, OUT_DIR)
  console.log(`[stage-agent-payloads] staging: site-packages (${target.key}, ${tag})`)
  stageSitePackages(target, OUT_DIR, payloadPython)
  // The glue that makes the payload interpreter resolve repo/ and
  // site-packages/ wherever the bundle sits. Written after both stages
  // exist so a failed staging run never leaves a .pth that points at
  // nothing.
  writeBundlePth(OUT_DIR, payloadPython)
  console.log(`[stage-agent-payloads] staging: node (${target.key}, ${tag})`)
  stageNode(target, OUT_DIR)

  const manifest = buildManifest({ tag, commit, target })
  fs.writeFileSync(path.join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n")
  console.log(`[stage-agent-payloads] wrote ${path.join(OUT_DIR, "manifest.json")}`)
}

if (isMain(import.meta.url)) {
  main()
}

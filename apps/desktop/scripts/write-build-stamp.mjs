"use strict"

/**
 * Writes apps/desktop/build/install-stamp.json with the git ref the desktop
 * .exe should pin to at first-launch bootstrap time.  This file ships inside
 * the packaged app via electron-builder's extraResources entry and is read
 * by electron/main.ts to drive the install.ps1 stage bootstrap flow.
 *
 * Schema (subject to bump via STAMP_SCHEMA_VERSION):
 *   {
 *     "schemaVersion": 1,
 *     "commit":        "<40-char SHA>",
 *     "branch":        "<branch name>",
 *     "repository":    "<github owner/repo>",
 *     "bootstrapRef":  "<remote ref used to fetch installer scripts>",
 *     "commitPinned":  true|false,
 *     "builtAt":       "<ISO 8601 UTC timestamp>",
 *     "dirty":         true|false,
 *     "source":        "ci" | "local"
 *   }
 *
 * Source preference order:
 *   1. CI env vars ($GITHUB_SHA / $GITHUB_REF_NAME) -- avoid edge cases with
 *      shallow clones, detached HEADs, etc. in CI.
 *   2. Local `git rev-parse` against the parent repo (../..).
 *
 * Dev / out-of-repo builds without git produce an explicit error rather than
 * silently writing an unstamped manifest -- the packaged app refuses to
 * bootstrap without a stamp.
 */

import { mkdirSync, writeFileSync } from "fs"
import { resolve, join, relative } from "path"
import { execFileSync, execSync } from "child_process"
import { isMain } from "./utils.mjs"

const STAMP_SCHEMA_VERSION = 1

const DESKTOP_ROOT = resolve(import.meta.dirname, "..")
const REPO_ROOT = resolve(DESKTOP_ROOT, "..", "..")
const OUT_DIR = join(DESKTOP_ROOT, "build")
const OUT_FILE = join(OUT_DIR, "install-stamp.json")

function tryExec(cmd, opts) {
  try {
    return execSync(cmd, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], ...opts }).trim()
  } catch {
    return null
  }
}

function tryGit(args) {
  try {
    return execFileSync("git", args, { cwd: REPO_ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim()
  } catch {
    return null
  }
}

function normalizeGitHubRepository(value) {
  if (!value || typeof value !== "string") return null
  const trimmed = value.trim()
  const sshMatch = trimmed.match(/^git@github\.com:([^/\s]+)\/([^/\s]+?)(?:\.git)?$/i)
  if (sshMatch) return `${sshMatch[1]}/${sshMatch[2]}`
  const httpsMatch = trimmed.match(/^https:\/\/github\.com\/([^/\s]+)\/([^/\s]+?)(?:\.git)?(?:\/)?$/i)
  if (httpsMatch) return `${httpsMatch[1]}/${httpsMatch[2]}`
  const slugMatch = trimmed.match(/^([^/\s]+)\/([^/\s]+)$/)
  if (slugMatch) return `${slugMatch[1]}/${slugMatch[2].replace(/\.git$/i, "")}`
  return null
}

function repoUrls(repository) {
  if (!repository) return { repoUrlHttps: null, repoUrlSsh: null }
  return {
    repoUrlHttps: `https://github.com/${repository}.git`,
    repoUrlSsh: `git@github.com:${repository}.git`
  }
}

function remoteNameFromRef(ref) {
  if (!ref || typeof ref !== "string" || !ref.includes("/")) return null
  return ref.split("/")[0]
}

function remoteBranchName(ref) {
  if (!ref || typeof ref !== "string" || !ref.includes("/")) return null
  return ref.split("/").slice(1).join("/")
}

function remoteBranchesContaining(commit) {
  const output = tryGit(["branch", "-r", "--contains", commit])
  if (!output) return []
  return output
    .split(/\r?\n/)
    .map(line => line.replace(/^\*\s*/, "").trim())
    .filter(line => line && !line.endsWith("/HEAD") && !line.includes(" -> "))
}

function remoteBranchExists(remoteName, branch) {
  if (!remoteName || !branch) return false
  return Boolean(tryGit(["show-ref", "--verify", `refs/remotes/${remoteName}/${branch}`]))
}

function localRemoteMetadata(commit, branch) {
  const upstream = tryGit(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
  const containingBranches = remoteBranchesContaining(commit)
  const remoteRef = containingBranches[0] || (upstream && remoteBranchExists(remoteNameFromRef(upstream), remoteBranchName(upstream)) ? upstream : null)
  const remoteName = remoteNameFromRef(remoteRef) || remoteNameFromRef(upstream) || "origin"
  const remoteUrl = tryGit(["remote", "get-url", remoteName]) || tryGit(["remote", "get-url", "origin"])
  const repository = normalizeGitHubRepository(remoteUrl) || "NousResearch/hermes-agent"
  const remoteBranch = remoteBranchName(remoteRef) || (remoteBranchExists(remoteName, branch) ? branch : "main")
  const commitPinned = containingBranches.length > 0
  return {
    repository,
    bootstrapRef: commitPinned ? commit : remoteBranch,
    commitPinned,
    ...repoUrls(repository)
  }
}

function fromCI() {
  const sha = process.env.GITHUB_SHA
  if (!sha) return null
  const branch = process.env.GITHUB_REF_NAME || process.env.GITHUB_HEAD_REF || null
  const repository = normalizeGitHubRepository(process.env.GITHUB_REPOSITORY) || "NousResearch/hermes-agent"
  return {
    commit: sha,
    branch: branch,
    repository,
    bootstrapRef: sha,
    commitPinned: true,
    ...repoUrls(repository),
    dirty: false, // CI builds from a checkout-of-ref by definition
    source: "ci"
  }
}

function fromLocalGit() {
  const sha = tryExec("git rev-parse HEAD", { cwd: REPO_ROOT })
  if (!sha) return null
  const branch = tryExec("git rev-parse --abbrev-ref HEAD", { cwd: REPO_ROOT })
  // `git status --porcelain -uno` is empty iff tracked files match HEAD.
  // We exclude untracked files (-uno) intentionally: a developer who's
  // checked out an installer scratch dir alongside the repo shouldn't
  // poison every local build with a [DIRTY] stamp.  We DO care about
  // tracked-but-modified files because those mean the .exe content
  // differs from the commit being pinned.
  const status = tryExec("git status --porcelain -uno", { cwd: REPO_ROOT })
  const dirty = status !== null && status.length > 0
  const normalizedBranch = branch === "HEAD" ? null : branch
  return {
    commit: sha,
    branch: normalizedBranch, // detached HEAD -> null
    ...localRemoteMetadata(sha, normalizedBranch),
    dirty: dirty,
    source: "local"
  }
}

function main() {
  const stamp = fromCI() || fromLocalGit()
  if (!stamp || !stamp.commit) {
    console.error(
      "[write-build-stamp] ERROR: could not determine git commit.\n" +
        "  - $GITHUB_SHA not set\n" +
        "  - `git rev-parse HEAD` failed at " +
        REPO_ROOT +
        "\n" +
        "Packaged builds require a git ref to pin first-launch install.ps1\n" +
        "against. Run from a git checkout or set $GITHUB_SHA explicitly."
    )
    process.exit(1)
  }

  if (stamp.dirty) {
    console.warn(
      "[write-build-stamp] WARNING: working tree is dirty.\n" +
        "  Pinning to " +
        stamp.commit.slice(0, 12) +
        " but the packaged code may differ from that commit.\n" +
        "  Commit your changes before publishing this build."
    )
  }

  if (stamp.source === "local" && stamp.commitPinned === false) {
    console.warn(
      "[write-build-stamp] WARNING: local HEAD is not contained in a fetched remote branch.\n" +
        "  The packaged app will bootstrap Hermes from " +
        stamp.repository +
        "@" +
        stamp.bootstrapRef +
        " instead of pinning unreachable commit " +
        stamp.commit.slice(0, 12) +
        ".\n" +
        "  Push the branch before publishing a release build."
    )
  }

  const payload = {
    schemaVersion: STAMP_SCHEMA_VERSION,
    commit: stamp.commit,
    branch: stamp.branch,
    repository: stamp.repository,
    bootstrapRef: stamp.bootstrapRef,
    commitPinned: stamp.commitPinned,
    repoUrlHttps: stamp.repoUrlHttps,
    repoUrlSsh: stamp.repoUrlSsh,
    builtAt: new Date().toISOString(),
    dirty: stamp.dirty,
    source: stamp.source
  }

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(OUT_FILE, JSON.stringify(payload, null, 2) + "\n", "utf8")
  console.log(
    "[write-build-stamp] wrote " +
      relative(REPO_ROOT, OUT_FILE) +
      " -> " +
      stamp.commit.slice(0, 12) +
      (stamp.branch ? " (" + stamp.branch + ")" : "") +
      (stamp.repository ? " [" + stamp.repository + "@" + stamp.bootstrapRef + "]" : "") +
      (stamp.commitPinned === false ? " [UNPINNED]" : "") +
      (stamp.dirty ? " [DIRTY]" : "")
  )
}

if (isMain(import.meta.url)) {
  main()
}

export { localRemoteMetadata, normalizeGitHubRepository, remoteBranchName, remoteNameFromRef, repoUrls }

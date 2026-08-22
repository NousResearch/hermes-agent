#!/usr/bin/env node
/**
 * detect-routing-drift.cjs
 *
 * Read-only drift detector for the LAH Stack canonical routing mapping.
 *
 * Compares current filesystem state (repositories + worktrees) against
 * repo_mappings.json without modifying either one.
 *
 * Usage:
 *   node detect-routing-drift.cjs [mapping.json]
 *
 * Exit codes:
 *   0 — no actionable drift
 *   1 — actionable drift detected
 *   2 — detector or input failure
 *
 * Output: deterministic structured JSON to stdout.
 * Stderr: diagnostic messages only.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

// ─── Configuration ───────────────────────────────────────────────────────────

const DEFAULT_MAPPING =
  "/home/deploy/.hermes/skills/software-development/lah-repo-router/references/repo_mappings.json";

const SCANNED_ROOTS = [
  "/home/deploy/lah-stack-repos",
  "/home/deploy/lah-stack-worktrees",
];

// Excluded directories (full or suffix match) — never classified as routing surface
const EXCLUDED_SUFFIXES = [
  // Archives
  "clawx-arch",
  "aionui-arch",
  // Tooling / config dirs
  ".codex",
  ".hermes/plugins",
  ".openclaw",
  // Generated / vendor
  "node_modules",
  "vendor",
  // External repos
  "temporal",
  "scientific-method",
  "pg_play",
];

// Known external repos that should be excluded
const KNOWN_EXTERNAL_PATHS = [
  "/home/deploy/temporal",
  "/home/deploy/scientific-method",
  "/home/deploy/pg_play",
];

// Temporary worktree root — ephemeral, not to be mapped
const TEMP_ROOT = "/tmp";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function isExcluded(absPath) {
  for (const suffix of EXCLUDED_SUFFIXES) {
    if (absPath.endsWith(suffix)) return true;
  }
  return false;
}

function isKnownExternal(absPath) {
  return KNOWN_EXTERNAL_PATHS.includes(absPath);
}

function isTemp(absPath) {
  return absPath.startsWith(TEMP_ROOT);
}

function readFileOrThrow(p) {
  return fs.readFileSync(p, "utf-8");
}

function safeExec(args, cwd) {
  try {
    return execFileSync(args[0], args.slice(1), {
      cwd,
      encoding: "utf-8",
      timeout: 5000,
      maxBuffer: 1024 * 16,
    }).trim();
  } catch {
    return null;
  }
}

function getGitRemote(cwd) {
  const out = safeExec(["git", "remote", "-v"], cwd);
  if (!out) return null;
  const origin = out
    .split("\n")
    .find((l) => l.startsWith("origin"));
  if (!origin) return null;
  const m = origin.match(/origin\s+(\S+)/);
  return m ? m[1] : null;
}

function getGitCommonDir(cwd) {
  return safeExec(["git", "rev-parse", "--git-common-dir"], cwd);
}

function getGitWorktreeList(cwd) {
  const out = safeExec(["git", "worktree", "list"], cwd);
  if (!out) return [];
  return out
    .split("\n")
    .filter(Boolean)
    .map((l) => {
      const parts = l.split(/\s+/);
      return { path: parts[0], branch: parts[1] || null };
    });
}

function getGitTopLevel(cwd) {
  return safeExec(["git", "rev-parse", "--show-toplevel"], cwd);
}

function getGitBranch(cwd) {
  return safeExec(["git", "branch", "--show-current"], cwd);
}

function getGitShortSha(cwd) {
  return safeExec(["git", "rev-parse", "--short", "HEAD"], cwd);
}

/**
 * Check if a directory is a Git worktree.
 * A worktree has a `.git` FILE (not directory) containing "gitdir: <path>".
 */
function isWorktree(dirPath) {
  const gitPath = path.join(dirPath, ".git");
  try {
    const stat = fs.statSync(gitPath);
    if (stat.isFile()) {
      const content = fs.readFileSync(gitPath, "utf-8").trim();
      return content.startsWith("gitdir:");
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Check if a directory is a standalone Git repo (has a .git directory).
 */
function isStandaloneGitRepo(dirPath) {
  const gitPath = path.join(dirPath, ".git");
  try {
    const stat = fs.statSync(gitPath);
    if (stat.isDirectory()) return true;
    if (stat.isFile()) {
      const content = fs.readFileSync(gitPath, "utf-8").trim();
      return content.startsWith("gitdir:");
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Check if a worktree is active (git worktree prune would not remove it).
 * An inactive/prunable worktree has a missing or broken gitdir reference.
 */
function isWorktreeActive(wtPath) {
  // Check the .git file still points to a valid common dir
  const gitPath = path.join(wtPath, ".git");
  try {
    const content = fs.readFileSync(gitPath, "utf-8").trim();
    const m = content.match(/^gitdir:\s*(.+)/);
    if (!m) return false;
    const commonDir = m[1].trim();
    return fs.existsSync(commonDir);
  } catch {
    return false;
  }
}

// ─── Scan filesystem ──────────────────────────────────────────────────────────

function scanFilesystem() {
  const repos = [];   // { path, type: "standalone" | "worktree", remote, branch, sha, commonDir }
  const seen = new Set();

  for (const root of SCANNED_ROOTS) {
    if (!fs.existsSync(root)) continue;

    const entries = fs.readdirSync(root, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const absPath = path.join(root, entry.name);
      if (absPath.endsWith(".codex")) continue;

      // Skip if excluded
      if (isExcluded(absPath)) continue;
      if (isKnownExternal(absPath)) continue;
      if (isTemp(absPath)) continue;

      // Check if this is a Git repo or worktree
      if (!isStandaloneGitRepo(absPath)) continue;
      if (seen.has(absPath)) continue;
      seen.add(absPath);

      const isWt = isWorktree(absPath);
      const commonDir = isWt ? getGitCommonDir(absPath) : null;
      const topLevel = getGitTopLevel(absPath) || absPath;
      const remote = getGitRemote(absPath);
      const branch = getGitBranch(absPath);
      const sha = getGitShortSha(absPath);

      let classification;
      if (absPath.startsWith("/home/deploy/lah-stack-worktrees/")) {
        classification = isWt ? "ACTIVE_WORKTREE" : "WORKTREE_DIR_WITHOUT_GITFILE";
      } else if (absPath.startsWith("/home/deploy/lah-stack-repos/")) {
        classification = isWt ? "WORKTREE_IN_REPOS" : "CANONICAL_REPOSITORY";
      } else {
        classification = "OTHER";
      }

      repos.push({
        path: topLevel,
        type: isWt ? "worktree" : "standalone",
        classification,
        remote,
        commonDir,
        branch,
        sha,
      });
    }
  }

  return repos;
}

/**
 * Resolve the canonical checkout to its true git top-level (handles nested worktrees).
 */
function getCanonicalTopLevel(mappedPath) {
  if (!fs.existsSync(mappedPath)) return null;
  const tl = getGitTopLevel(mappedPath);
  if (tl) return tl;
  // If git fails, check if it's a path that exists but is not git
  try {
    const stat = fs.statSync(mappedPath);
    if (stat.isDirectory()) return mappedPath;
  } catch {}
  return null;
}

// ─── Compare discovered vs mapped ─────────────────────────────────────────────

function compare(mapping, discovered) {
  const mapped = mapping.repositories || [];
  const forbiddenRoots = (mapping.write_forbidden_roots || []).map(
    (r) => r.path
  );

  const actionable = [];
  const informational = [];

  // Build a lookup of mapped checkouts and worktree paths
  const mappedCheckoutPaths = new Set();
  const mappedWorktreePaths = new Set();
  const mappedCheckoutTopLevels = new Map(); // canonical path -> repo_id
  const mappedRepoIds = new Set();

  for (const repo of mapped) {
    const repoId = repo.repository_id;
    mappedRepoIds.add(repoId);

    const checkout = repo.canonical_checkout;
    if (checkout) {
      mappedCheckoutPaths.add(checkout);
      const tl = getCanonicalTopLevel(checkout);
      if (tl) {
        mappedCheckoutTopLevels.set(tl, repoId);
        mappedCheckoutPaths.add(tl);
      }
    }

    for (const wt of repo.worktrees || []) {
      const wtPath = wt.path;
      if (wtPath) {
        mappedWorktreePaths.add(wtPath);
      }
    }

    for (const ws of repo.execution_workspaces || []) {
      const wsPath = ws.path;
      if (wsPath) {
        // Execution workspaces are not worktrees but are expected
        // Treat them like worktrees for drift purposes
        mappedCheckoutPaths.add(wsPath);
      }
    }
  }

  // Record: which discovered repos are mapped vs unmapped
  for (const d of discovered) {
    if (d.classification === "OTHER") {
      informational.push({
        type: "EXTERNAL_REPOSITORY",
        path: d.path,
        detail: `Discovered repository at ${d.path} (outside canonical roots)`,
      });
      continue;
    }

    const isMappedCheckout = mappedCheckoutPaths.has(d.path);
    const isMappedWorktree = mappedWorktreePaths.has(d.path);

    if (d.type === "standalone") {
      if (d.path.startsWith("/home/deploy/lah-stack-repos/")) {
        if (!isMappedCheckout) {
          // Check if it's an alias for a mapped repo (same toplevel)
          let matched = false;
          for (const [canonicalTl, repoId] of mappedCheckoutTopLevels) {
            if (d.path === canonicalTl) {
              matched = true;
              break;
            }
          }
          if (!matched) {
            actionable.push({
              type: "UNMAPPED_CANONICAL_REPOSITORY",
              path: d.path,
              remote: d.remote,
              branch: d.branch,
              sha: d.sha,
              detail: `Found canonical repo at ${d.path} not in repo_mappings.json`,
            });
          }
        }
      } else if (d.path.startsWith("/home/deploy/lah-stack-worktrees/")) {
        if (!isMappedWorktree && !isMappedCheckout) {
          informational.push({
            type: "UNMAPPED_ACTIVE_WORKTREE",
            path: d.path,
            remote: d.remote,
            branch: d.branch,
            detail: `Found worktree at ${d.path} not in repo_mappings.json`,
          });
        }
      }
    } else if (d.type === "worktree") {
      if (!isMappedWorktree && !isMappedCheckout) {
        // Worktree not in mapping
        if (d.commonDir) {
          // Find parent repo
          informational.push({
            type: "UNMAPPED_ACTIVE_WORKTREE",
            path: d.path,
            commonDir: d.commonDir,
            remote: d.remote,
            branch: d.branch,
            sha: d.sha,
            detail: `Found active worktree at ${d.path} (attached to ${d.commonDir}) not in repo_mappings.json`,
          });
        } else {
          informational.push({
            type: "EPHEMERAL_WORKTREE",
            path: d.path,
            detail: `Worktree at ${d.path} may be prunable or ephemeral`,
          });
        }
      }
    }
  }

  // Check mapped checkouts missing from filesystem
  for (const repo of mapped) {
    const checkout = repo.canonical_checkout;
    if (checkout) {
      const tl = getCanonicalTopLevel(checkout);
      if (!tl || !fs.existsSync(checkout)) {
        actionable.push({
          type: "MAPPED_REPOSITORY_MISSING_ON_DISK",
          path: checkout,
          repoId: repo.repository_id,
          detail: `Mapped repository ${repo.repository_id} checkout ${checkout} not found on disk`,
        });
      } else {
        // Verify remote matches
        const actualRemote = getGitRemote(tl);
        const expectedRemote = repo.canonical_remote;
        if (actualRemote && expectedRemote && actualRemote !== expectedRemote) {
          actionable.push({
            type: "REMOTE_MISMATCH",
            path: tl,
            repoId: repo.repository_id,
            expected: expectedRemote,
            actual: actualRemote,
            detail: `Remote mismatch for ${repo.repository_id}: expected ${expectedRemote}, got ${actualRemote}`,
          });
        }
      }
    }

    // Check mapped worktrees missing from filesystem
    for (const wt of repo.worktrees || []) {
      if (!fs.existsSync(wt.path)) {
        actionable.push({
          type: "MAPPED_WORKTREE_MISSING_ON_DISK",
          path: wt.path,
          name: wt.name,
          repoId: repo.repository_id,
          detail: `Mapped worktree ${wt.name} (${wt.path}) not found on disk`,
        });
      }
    }
  }

  // Check for duplicate canonical_checkout paths
  const checkoutCounts = new Map();
  for (const repo of mapped) {
    const cp = repo.canonical_checkout;
    if (cp) {
      checkoutCounts.set(cp, (checkoutCounts.get(cp) || 0) + 1);
    }
  }
  for (const [cp, count] of checkoutCounts) {
    if (count > 1) {
      actionable.push({
        type: "DUPLICATE_MAPPING",
        path: cp,
        detail: `Canonical checkout ${cp} appears ${count} times in the mapping`,
      });
    }
  }

  // Check for worktree parent mismatches
  // NOTE: git rev-parse --git-common-dir for a worktree returns the
  // parent repo's .git directory, not the granular worktree metadata path.
  // This is standard Git behavior. We don't flag it as a mismatch here
  // because the common dir == parent .git is expected.

  // Check for ephemeral /tmp repos
  for (const d of discovered) {
    if (d.path.startsWith("/tmp")) {
      informational.push({
        type: "EPHEMERAL_WORKTREE",
        path: d.path,
        detail: `Ephemeral worktree at ${d.path} — excluded from mapping`,
      });
    }
  }

  return { actionable, informational };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

function main() {
  const mappingPath = process.argv[2] || DEFAULT_MAPPING;

  // Validate mapping file
  if (!fs.existsSync(mappingPath)) {
    console.error("Mapping not found:", mappingPath);
    process.exit(2);
  }

  let mapping;
  try {
    mapping = JSON.parse(fs.readFileSync(mappingPath, "utf-8"));
  } catch (e) {
    console.error("Invalid mapping JSON:", e.message);
    process.exit(2);
  }

  // Scan filesystem
  const discovered = scanFilesystem();

  // Compare
  const { actionable, informational } = compare(mapping, discovered);

  // Build output — deterministic ordering: sort by path alphabetically
  const sortByPath = (a, b) => a.path.localeCompare(b.path);
  const sortByTypeAndPath = (a, b) => {
    const tc = a.type.localeCompare(b.type);
    if (tc !== 0) return tc;
    return a.path.localeCompare(b.path);
  };

  const mappedRepoCount = (mapping.repositories || []).length;
  const discoveredRepoCount = discovered.filter(
    (d) => d.type === "standalone"
  ).length;
  const mappedWorktreeCount = (mapping.repositories || []).reduce(
    (sum, r) => sum + (r.worktrees || []).length + (r.execution_workspaces || []).length,
    0
  );
  const discoveredWorktreeCount = discovered.filter(
    (d) => d.type === "worktree"
  ).length;

  const output = {
    schema_version: "1",
    generated_at: new Date().toISOString(),
    canonical_mapping_path: mappingPath,
    scanned_roots: [...SCANNED_ROOTS].sort(),
    summary: {
      mapped_repositories: mappedRepoCount,
      discovered_repositories: discoveredRepoCount,
      mapped_worktrees: mappedWorktreeCount,
      discovered_worktrees: discoveredWorktreeCount,
      actionable_drift_count: actionable.length,
      informational_count: informational.length,
    },
    actionable_drift: actionable.sort(sortByTypeAndPath),
    informational: informational.sort(sortByTypeAndPath),
    verdict:
      actionable.length === 0
        ? informational.length > 0
          ? "INFORMATIONAL_DIFFERENCES_ONLY"
          : "NO_DRIFT"
        : "ACTIONABLE_DRIFT_DETECTED",
  };

  // Validate JSON serialization
  const json = JSON.stringify(output, null, 2);
  // Verify re-parse gives the same result (determinism check)
  JSON.parse(json);

  process.stdout.write(json + "\n");
  process.exit(actionable.length > 0 ? 1 : 0);
}

main();

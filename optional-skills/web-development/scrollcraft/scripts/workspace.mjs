#!/usr/bin/env node
/**
 * Resolve the scrollcraft workspace: the one directory that holds this user's
 * builds and their fingerprint registry.
 *
 * The skill used to hardcode one author's repo layout, which meant the
 * fingerprint gate dead-ended on every other machine. Nothing is hardcoded now.
 * Resolution order, first hit wins:
 *
 *   1. SCROLLCRAFT_HOME              env var, absolute or relative to cwd
 *   2. .scrollcraft.json             nearest one walking up from cwd,
 *                                    { "workspace": "some/path" } relative to
 *                                    the file that declares it
 *   3. <project root>/scrollcraft    project root = nearest ancestor with .git,
 *                                    otherwise cwd
 *
 * USAGE
 *   node scripts/workspace.mjs                 print the workspace path
 *   node scripts/workspace.mjs --json          print every resolved path
 *   node scripts/workspace.mjs --ensure        create it and seed the registry
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SKILL = path.resolve(HERE, "..");

function findUp(startDir, predicate) {
  let dir = path.resolve(startDir);
  for (let i = 0; i < 12; i++) {
    const hit = predicate(dir);
    if (hit) return { dir, hit };
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return null;
}

export function resolveWorkspace(cwd = process.cwd()) {
  if (process.env.SCROLLCRAFT_HOME) {
    return { root: path.resolve(cwd, process.env.SCROLLCRAFT_HOME), via: "SCROLLCRAFT_HOME" };
  }

  const cfg = findUp(cwd, (d) => {
    const p = path.join(d, ".scrollcraft.json");
    return fs.existsSync(p) ? p : null;
  });
  if (cfg) {
    let parsed = {};
    try {
      parsed = JSON.parse(fs.readFileSync(cfg.hit, "utf8"));
    } catch (e) {
      throw new Error(`.scrollcraft.json is not valid JSON: ${cfg.hit}\n  ${e.message}`);
    }
    if (parsed.workspace) {
      return { root: path.resolve(cfg.dir, parsed.workspace), via: cfg.hit };
    }
  }

  const git = findUp(cwd, (d) => (fs.existsSync(path.join(d, ".git")) ? d : null));
  const base = git ? git.hit : path.resolve(cwd);
  return { root: path.join(base, "scrollcraft"), via: git ? "project root (.git)" : "cwd" };
}

export function paths(cwd = process.cwd()) {
  const { root, via } = resolveWorkspace(cwd);
  return {
    via,
    workspace: root,
    builds: path.join(root, "builds"),
    lab: path.join(root, "lab"),
    fingerprints: path.join(root, "FINGERPRINTS.md"),
  };
}

export function ensure(cwd = process.cwd()) {
  const p = paths(cwd);
  fs.mkdirSync(p.builds, { recursive: true });
  fs.mkdirSync(p.lab, { recursive: true });
  let seeded = false;
  if (!fs.existsSync(p.fingerprints)) {
    const seed = path.join(SKILL, "templates", "FINGERPRINTS.md");
    fs.copyFileSync(seed, p.fingerprints);
    seeded = true;
  }
  return { ...p, seeded };
}

// ------------------------------------------------------------------- cli ----
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const argv = process.argv.slice(2);
  const p = argv.includes("--ensure") ? ensure() : paths();
  if (argv.includes("--json")) {
    console.log(JSON.stringify(p, null, 2));
  } else if (argv.includes("--ensure")) {
    console.log(p.workspace);
    console.error(`  builds       ${p.builds}`);
    console.error(`  registry     ${p.fingerprints}${p.seeded ? "  (seeded, empty)" : "  (already present)"}`);
    console.error(`  resolved via ${p.via}`);
  } else {
    console.log(p.workspace);
  }
}

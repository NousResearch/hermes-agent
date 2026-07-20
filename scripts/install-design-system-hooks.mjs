#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const registryPath = path.resolve(root, "docs/design/dashboard-kit-adoption.json");
const hookName = "pre-commit";
const hookStart = "# >>> hermes-dashboard-design-system";
const hookEnd = "# <<< hermes-dashboard-design-system";

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function findRepoRoot(start) {
  let current = start;
  while (current && current !== path.dirname(current)) {
    if (fs.existsSync(path.join(current, ".git"))) return current;
    current = path.dirname(current);
  }
  return null;
}

function unique(values) {
  return [...new Set(values)];
}

function hookBlock() {
  return `${hookStart}
cd "${root}" || exit 1
if npm run dashboard:design-system:status -- --strict; then
  exit 0
fi

echo "Hermes dashboard design-system drift detected. Attempting auto-heal..."
npm run dashboard:design-system:status -- --sync
npm run dashboard:design-system:status -- --strict

echo ""
echo "Hermes dashboard CSS adapters were auto-healed."
echo "Review and stage the synced dashboard CSS files, then commit again."
exit 1
${hookEnd}`;
}

function installHook(repoRoot) {
  const gitPath = path.join(repoRoot, ".git");
  const hooksDir = path.join(gitPath, "hooks");
  const hookPath = path.join(hooksDir, hookName);

  if (!fs.existsSync(gitPath) || !fs.statSync(gitPath).isDirectory()) {
    return { repoRoot, status: "skipped", reason: ".git is not a directory" };
  }

  fs.mkdirSync(hooksDir, { recursive: true });

  const existing = fs.existsSync(hookPath) ? fs.readFileSync(hookPath, "utf8") : "#!/bin/sh\n";
  const cleaned = existing
    .replace(new RegExp(`\\n?${hookStart}[\\s\\S]*?${hookEnd}\\n?`, "g"), "\n")
    .trimEnd();
  const next = `${cleaned}\n\n${hookBlock()}\n`;

  fs.writeFileSync(hookPath, next, { mode: 0o755 });
  fs.chmodSync(hookPath, 0o755);
  return { repoRoot, status: "installed", reason: hookPath };
}

if (!fs.existsSync(registryPath)) {
  console.error(`Missing adoption registry: ${registryPath}`);
  process.exit(1);
}

const registry = readJson(registryPath);
const targetRoots = registry.dashboards
  .map((dashboard) => path.resolve(root, dashboard.targetPath))
  .map((target) => findRepoRoot(path.dirname(target)))
  .filter(Boolean);

const repoRoots = unique([root, ...targetRoots]);
const results = repoRoots.map(installHook);

console.table(
  results.map((result) => ({
    repo: path.relative(path.dirname(root), result.repoRoot) || path.basename(result.repoRoot),
    status: result.status,
    detail: result.reason,
  })),
);

const failed = results.filter((result) => result.status !== "installed");
if (failed.length) {
  console.error(`Some hooks were not installed (${failed.length}).`);
  process.exit(1);
}

console.log(`Installed design-system pre-commit hooks in ${results.length} repositories.`);

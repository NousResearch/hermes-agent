#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("trackable V8-V14 plan", () => {
  const text = read("docs/design/v8-v14-trackable-build-plan.md");
  for (const version of ["V8", "V9", "V10", "V11", "V12", "V13", "V14"]) {
    requireIncludes(text, `## ${version}:`, `${version} section`);
  }
  requireIncludes(text, "Media Engine Ops Migration", "Media Engine migration");
  requireIncludes(text, "Khashi VC ROC Migration", "Khashi migration");
});

check("V8 migration route", () => {
  const route = read("web/src/dashboard-route-registry.tsx");
  requireIncludes(route, '"/dashboard-migrations": PackageNativeMigrationsPage', "route registry entry");
  requireIncludes(route, '"/package-native/media-engine": MediaEnginePackageNativePage', "Media Engine package-native route");
  requireIncludes(route, '"/package-native/khashi-vc": KhashiVcPackageNativePage', "Khashi package-native route");
  requireIncludes(route, "Dashboard Migrations", "nav entry");
  requireIncludes(route, "Media Native", "Media package-native nav entry");
  requireIncludes(route, "Khashi Native", "Khashi package-native nav entry");
  const page = read("web/src/pages/PackageNativeMigrationsPage.tsx");
  for (const recipe of ["pipeline-workflow-dashboard", "operations-control-room", "market-asset-explorer", "executive-command-center"]) {
    requireIncludes(page, recipe, `recipe ${recipe}`);
  }
  requireIncludes(page, "Retire Adapter", "adapter retirement guardrail");
  requireIncludes(page, "/package-native/media-engine", "Media Engine package-native migration link");
  requireIncludes(page, "/package-native/khashi-vc", "Khashi package-native migration link");
});

check("V8 package-native shadow dashboards", () => {
  const page = read("web/src/pages/ProjectSnapshotDashboardPage.tsx");
  for (const phrase of [
    "MediaEnginePackageNativePage",
    "KhashiVcPackageNativePage",
    "DashboardSnapshot",
    "DashboardShell",
    "DataTable",
    "KpiCard",
    "Adapter Retirement Gate",
  ]) {
    requireIncludes(page, phrase, `package-native dashboard ${phrase}`);
  }
  const metadata = read("web/src/dashboard-page-metadata.ts");
  requireIncludes(metadata, 'route: "/package-native/media-engine"', "Media Engine package-native metadata");
  requireIncludes(metadata, 'route: "/package-native/khashi-vc"', "Khashi package-native metadata");
  requireIncludes(metadata, "Shadow package-native route used for V8 parity", "shadow route local-only reason");
});

check("V8 local visual coverage exists", () => {
  const tests = read("tests/dashboard/design-system.spec.ts");
  const config = read("playwright.dashboard.config.ts");
  for (const phrase of [
    "/package-native/media-engine",
    "/package-native/khashi-vc",
    "Media Engine package-native shadow dashboard preserves V8 signal surface",
    "Khashi VC package-native shadow dashboard preserves V8 signal surface",
    "media-engine-package-native",
    "khashi-vc-package-native",
  ]) {
    requireIncludes(tests, phrase, `V8 visual coverage ${phrase}`);
  }
  requireIncludes(config, 'testIgnore: "v8-production-cutover.spec.ts"', "local visual suite excludes production cutover test");
});

check("V8 production cutover check exists", () => {
  const config = read("playwright.v8-production.config.ts");
  const test = read("tests/dashboard/v8-production-cutover.spec.ts");
  for (const phrase of [
    "HERMES_AGENT_PRODUCTION_URL",
    "v8-production-cutover.spec.ts",
    "desktop",
    "mobile",
  ]) {
    requireIncludes(config, phrase, `production Playwright config ${phrase}`);
  }
  for (const phrase of [
    "/api/status",
    "/package-native/media-engine",
    "/package-native/khashi-vc",
    "HERMES_AGENT_DASHBOARD_USERNAME",
    "HERMES_AGENT_DASHBOARD_PASSWORD",
    "Adapter Retirement Gate",
  ]) {
    requireIncludes(test, phrase, `production cutover test ${phrase}`);
  }
});

check("V8 cutover and rollback evidence is documented", () => {
  const cutover = read("docs/design/v8-dashboard-cutover-evidence.md");
  const rollback = read("docs/design/v8-static-adapter-rollback-plan.md");
  for (const phrase of [
    "Media Engine Ops",
    "Khashi VC ROC",
    "Production Evidence Still Required",
    "retirementAllowed",
  ]) {
    requireIncludes(cutover, phrase, `cutover evidence ${phrase}`);
  }
  for (const phrase of [
    "Rollback Steps",
    "Media Engine Ops",
    "Khashi VC ROC",
    "Recovery Rule",
  ]) {
    requireIncludes(rollback, phrase, `rollback plan ${phrase}`);
  }
});

check("package-native parity registry", () => {
  const registry = JSON.parse(read("docs/design/package-native-parity-registry.json"));
  if (!Array.isArray(registry.targets) || registry.targets.length < 3) {
    throw new Error("expected at least three package-native migration targets");
  }
  for (const target of registry.targets) {
    for (const field of ["id", "dashboard", "currentSurface", "targetSurface", "recipe", "adapterPath", "parity", "nextStep"]) {
      if (!target[field]) throw new Error(`${target.id || "target"} missing ${field}`);
    }
    if (["media-engine.ops", "khashi-vc.roc"].includes(target.id)) {
      if (!target.packageNativeRoute) throw new Error(`${target.id} missing packageNativeRoute`);
      if (target.parity.packageNativeShadowRoute !== true) throw new Error(`${target.id} missing packageNativeShadowRoute parity`);
      if (target.parity.playwrightCoverage !== true) throw new Error(`${target.id} missing local Playwright coverage parity`);
      if (target.parity.rollbackPath !== true) throw new Error(`${target.id} missing rollback path parity`);
    }
    if (target.retirementAllowed) {
      for (const [key, value] of Object.entries(target.parity)) {
        if (value !== true) throw new Error(`${target.id} allows retirement without ${key}`);
      }
    }
  }
});

check("priority project snapshot endpoints are implemented", () => {
  const externalFiles = [
    ["../media-engine/tasks/ops-dashboard-server.js", "Media Engine snapshot route"],
    ["../khashi-vc/src/web/server.ts", "Khashi snapshot route"],
  ];
  const missing = externalFiles.filter(([relativePath]) => !exists(relativePath));
  if (missing.length && process.env.CI === "true") {
    console.log(`skip external workspace endpoint check in CI: ${missing.map(([relativePath]) => relativePath).join(", ")}`);
    return;
  }
  for (const [relativePath, label] of externalFiles) {
    requireIncludes(read(relativePath), "/dashboard-snapshot", label);
  }
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v8:validate"] !== "node scripts/validate-v8-dashboard-migrations.mjs") {
    throw new Error("missing dashboard:v8:validate script");
  }
  if (pkg.scripts?.["dashboard:v8:production:check"] !== "npx playwright test -c playwright.v8-production.config.ts") {
    throw new Error("missing dashboard:v8:production:check script");
  }
});

check("production image rebuilds on dashboard changes", () => {
  const workflow = read(".github/workflows/docker-publish.yml");
  for (const watchedPath of [
    "web/**",
    "ui-tui/**",
    "packages/hermes-dashboard-kit/**",
    "package-lock.json",
    ".dockerignore",
  ]) {
    requireIncludes(workflow, watchedPath, `Docker publish watches ${watchedPath}`);
  }
  const dockerignore = read(".dockerignore");
  requireIncludes(dockerignore, "hermes_cli/web_dist/", "local web_dist excluded from Docker context");
});

finish("V8 dashboard migration validation");

function check(label, fn) {
  try {
    fn();
    console.log(`ok ${label}`);
  } catch (error) {
    failures.push(`${label}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function finish(label) {
  if (failures.length) {
    console.error(`${label} failed (${failures.length})`);
    for (const failure of failures) console.error(`- ${failure}`);
    process.exit(1);
  }
  console.log(`${label} passed`);
}

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function exists(relativePath) {
  return fs.existsSync(path.join(root, relativePath));
}

function requireIncludes(text, needle, label) {
  if (!text.includes(needle)) throw new Error(`missing ${label}`);
}

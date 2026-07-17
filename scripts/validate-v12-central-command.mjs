#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("central command route", () => {
  requireIncludes(read("web/src/dashboard-route-registry.tsx"), '"/central-command": CentralCommandPage', "central command route");
  requireIncludes(read("web/src/dashboard-page-metadata.ts"), 'route: "/central-command"', "central command metadata");
});

check("central command data", () => {
  const data = read("web/src/pages/central-command-data.ts");
  for (const phrase of ["DailyBriefItem", "CentralCommandData", "buildCentralCommandData", "buildKnownDashboardSnapshots", "ActionNeeded"]) {
    requireIncludes(data, phrase, phrase);
  }
});

check("central command page", () => {
  const page = read("web/src/pages/CentralCommandPage.tsx");
  for (const phrase of ["Hermes Central Command", "Daily Cross-Project Brief", "Business Impact Read", "ExecutiveActionQueue"]) {
    requireIncludes(page, phrase, phrase);
  }
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v12:validate"] !== "node scripts/validate-v12-central-command.mjs") {
    throw new Error("missing dashboard:v12:validate script");
  }
});

finish("V12 central command validation");

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

function requireIncludes(text, needle, label) {
  if (!text.includes(needle)) throw new Error(`missing ${label}`);
}

#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("theme package exports", () => {
  const themes = read("packages/hermes-dashboard-kit/src/themes.ts");
  for (const phrase of ["DashboardThemeProfile", "DashboardThemeTokenSet", "dashboardThemeProfiles", "tlc-base", "khashi-research", "media-publishing", "business-analytics"]) {
    requireIncludes(themes, phrase, phrase);
  }
  requireIncludes(read("packages/hermes-dashboard-kit/src/index.ts"), 'export * from "./themes"', "theme export");
});

check("theme system route", () => {
  requireIncludes(read("web/src/dashboard-route-registry.tsx"), '"/theme-system": ThemeSystemPage', "theme route");
  requireIncludes(read("web/src/dashboard-page-metadata.ts"), 'route: "/theme-system"', "theme metadata");
  const page = read("web/src/pages/ThemeSystemPage.tsx");
  requireIncludes(page, "Multi-Brand Dashboard Themes", "page heading");
  requireIncludes(page, "Token Swatches", "token swatches");
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v13:validate"] !== "node scripts/validate-v13-dashboard-themes.mjs") {
    throw new Error("missing dashboard:v13:validate script");
  }
});

finish("V13 dashboard themes validation");

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

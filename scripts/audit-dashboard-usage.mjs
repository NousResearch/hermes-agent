#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const strict = process.argv.includes("--strict");

const dashboardFiles = [
  { path: "web/src/pages/HermesOsPage.tsx", kind: "react-shell" },
  { path: "web/src/pages/DesignSystemPage.tsx", kind: "react-route" },
  { path: "../khashi-vc/public/roc/index.html", kind: "static-shell" },
  { path: "../khashi-vc/public/roc/app.js", kind: "static-runtime" },
  { path: "../media-engine/core/operations/unified-publishing-dashboard.js", kind: "static-shell" },
  { path: "../business-mapper/business_mapper/static/index.html", kind: "static-shell" },
  { path: "../business-mapper/business_mapper/static/app.js", kind: "static-runtime" },
  { path: "../media-business-operations/public/dashboard/index.html", kind: "static-shell" },
  { path: "../media-business-operations/public/dashboard/app.js", kind: "static-runtime" },
  { path: "../Meal-assistant/src/server.js", kind: "static-shell" },
];

const checks = [
  {
    id: "dashboard-shell",
    description: "dashboard shell primitives are present",
    appliesTo: ["react-shell", "static-shell"],
    requiredAny: ["DashboardShell", "hdk-shell", "hdk-body"],
    severity: "error",
  },
  {
    id: "metric-primitives",
    description: "metric primitives use kit or adapter classes",
    appliesTo: ["react-shell", "react-route", "static-shell", "static-runtime"],
    requiredAny: ["MetricGrid", "KpiCard", "hdk-metric-grid", "hdk-kpi-value", "hdk-card"],
    severity: "warning",
  },
  {
    id: "table-primitives",
    description: "table/list-heavy dashboards use table primitives or adapter wrappers",
    appliesTo: ["react-shell", "react-route", "static-shell", "static-runtime"],
    requiredAny: ["DataTable", "hdk-table", "hdk-table-wrap"],
    severity: "warning",
  },
  {
    id: "empty-state-primitives",
    description: "empty/error/loading states use shared primitives or adapter classes",
    appliesTo: ["react-shell", "react-route", "static-shell", "static-runtime"],
    requiredAny: ["DashboardEmptyState", "DashboardErrorState", "DashboardLoadingState", "hdk-empty", "hdk-error", "hdk-loading"],
    severity: "warning",
  },
  {
    id: "button-accessibility",
    description: "icon-only buttons should expose labels",
    appliesTo: ["react-shell", "react-route"],
    forbidden: [/<button(?![^>]*(aria-label|title|>\s*[^<\s]))[^>]*>\s*<[A-Z][^>]*\/?>\s*<\/button>/g],
    severity: "error",
  },
  {
    id: "hardcoded-colors",
    description: "dashboard code should avoid new hardcoded color values outside the adapter/theme layer",
    appliesTo: ["react-shell", "react-route"],
    forbidden: [/#(?:[0-9a-fA-F]{3}){1,2}\b/g, /rgba?\(/g],
    severity: "warning",
  },
];

const evidenceByCheck = {
  "metric-primitives": [
    "metric",
    "Metric",
    "kpi",
    "KPI",
    "summary",
    "total",
    "count",
    "hdk-metric-grid",
  ],
  "table-primitives": [
    "<table",
    "table",
    "Table",
    "table-row",
    "simpleTable",
    "data-table",
    "DataTable",
    "hdk-table",
  ],
  "empty-state-primitives": [
    "empty",
    "Empty",
    "loading",
    "Loading",
    "error",
    "Error",
    "unavailable",
    "hdk-empty",
  ],
};

function readFile(relativePath) {
  const absolutePath = path.resolve(root, relativePath);
  if (!fs.existsSync(absolutePath)) {
    return { relativePath, absolutePath, exists: false, text: "" };
  }
  return {
    relativePath,
    absolutePath,
    exists: true,
    text: fs.readFileSync(absolutePath, "utf8"),
  };
}

function lineFor(text, index) {
  return text.slice(0, index).split("\n").length;
}

function applies(check, kind) {
  return check.appliesTo.includes(kind);
}

function hasEvidenceForCheck(file, check) {
  const evidence = evidenceByCheck[check.id];
  if (!evidence) return true;
  return evidence.some((needle) => file.text.includes(needle));
}

function runRequiredCheck(file, check) {
  if (!check.requiredAny) return [];
  if (check.requiredAny.some((needle) => file.text.includes(needle))) return [];
  return [{
    severity: check.severity,
    check: check.id,
    file: file.relativePath,
    line: 1,
    message: `Missing ${check.description}. Expected one of: ${check.requiredAny.join(", ")}`,
  }];
}

function runForbiddenCheck(file, check) {
  if (!check.forbidden) return [];
  const findings = [];
  for (const pattern of check.forbidden) {
    for (const match of file.text.matchAll(pattern)) {
      findings.push({
        severity: check.severity,
        check: check.id,
        file: file.relativePath,
        line: lineFor(file.text, match.index ?? 0),
        message: `${check.description}: ${match[0].slice(0, 120)}`,
      });
    }
  }
  return findings;
}

function auditFile(file, kind) {
  if (!file.exists) {
    return [{
      severity: "warning",
      check: "file-present",
      file: file.relativePath,
      line: 1,
      message: "Configured dashboard file was not found; remove it from the audit list or add the surface.",
    }];
  }
  return checks
    .filter((check) => applies(check, kind) && hasEvidenceForCheck(file, check))
    .flatMap((check) => [
      ...runRequiredCheck(file, check),
      ...runForbiddenCheck(file, check),
    ]);
}

const files = dashboardFiles.map((entry) => ({ file: readFile(entry.path), kind: entry.kind }));

const findings = files.flatMap(({ file, kind }) => auditFile(file, kind));
const errors = findings.filter((finding) => finding.severity === "error");
const warnings = findings.filter((finding) => finding.severity === "warning");

console.log(`Dashboard usage audit: ${files.length} files checked`);
console.log(`Errors: ${errors.length}`);
console.log(`Warnings: ${warnings.length}`);

for (const finding of findings) {
  console.log(`${finding.severity.toUpperCase()} ${finding.file}:${finding.line} [${finding.check}] ${finding.message}`);
}

if (strict && findings.length) {
  process.exitCode = 1;
} else if (errors.length) {
  process.exitCode = 1;
}

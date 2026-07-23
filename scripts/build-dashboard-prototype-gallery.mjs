#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const registryPath = path.join(root, "docs/design/dashboard-prototype-registry.json");
const outputDir = path.join(root, "docs/design/prototype-gallery");
const outputPath = path.join(outputDir, "index.html");
const webDataPath = path.join(root, "web/src/pages/dashboard-prototype-data.ts");

const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"));
fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(outputPath, renderGallery(registry));
fs.writeFileSync(webDataPath, renderWebData(registry));
console.log(`Wrote ${path.relative(root, outputPath)}`);
console.log(`Wrote ${path.relative(root, webDataPath)}`);

function renderGallery(registry) {
  const sets = registry.prototypeSets ?? [];
  const generatedAt = new Date().toISOString();
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hermes Dashboard Prototype Gallery</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fa;
      --panel: #ffffff;
      --panel-soft: #f0f4f7;
      --text: #17202a;
      --muted: #5f6f7d;
      --border: #d7e0e7;
      --border-strong: #afbdc8;
      --primary: #176b87;
      --success: #2f7a53;
      --warning: #9b6b16;
      --critical: #a83a3a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    main { max-width: 1440px; margin: 0 auto; padding: 28px; }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 26px; font-weight: 720; }
    h2 { font-size: 18px; font-weight: 720; }
    h3 { font-size: 15px; font-weight: 700; }
    .muted { color: var(--muted); }
    .meta {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .set {
      margin-top: 22px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      overflow: hidden;
    }
    .set-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      padding: 18px;
      border-bottom: 1px solid var(--border);
      background: #fbfcfd;
    }
    .question-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .selected-line {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .chip {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 12px;
      color: var(--muted);
      background: var(--panel);
    }
    .variants {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      padding: 18px;
    }
    .variant {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      min-height: 430px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .variant-top {
      padding: 14px;
      border-bottom: 1px solid var(--border);
      background: var(--panel-soft);
    }
    .status {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--border-strong);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: var(--panel);
    }
    .workspace-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .workspace {
      border-radius: 6px;
      padding: 4px 7px;
      font-size: 12px;
      background: #dff0f3;
      color: #124f65;
    }
    .variant-body { padding: 14px; display: grid; gap: 14px; }
    .mini-section { display: grid; gap: 8px; }
    .mini-title {
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
      font-size: 11px;
      font-weight: 760;
    }
    ul { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }
    .requirement {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 8px;
      font-size: 13px;
    }
    .available { color: var(--success); }
    .partial, .unknown { color: var(--warning); }
    .missing { color: var(--critical); }
    .footer-note {
      margin-top: 22px;
      color: var(--muted);
      font-size: 13px;
    }
    @media (max-width: 1100px) {
      .variants { grid-template-columns: 1fr; }
      header, .set-header { grid-template-columns: 1fr; }
      .meta { white-space: normal; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Hermes Dashboard Prototype Gallery</h1>
        <p class="muted">Generated from the prototype registry. Use this to compare operator workflows before production UI changes.</p>
      </div>
      <div class="meta">${escapeHtml(sets.length)} prototype set${sets.length === 1 ? "" : "s"} · ${escapeHtml(generatedAt)}</div>
    </header>
    ${sets.map(renderSet).join("\n")}
    <p class="footer-note">Promotion rule: select a variant only after the data requirements, operator workflow, and six-workspace mapping are explicit.</p>
  </main>
</body>
</html>`;
}

function renderSet(set) {
  return `<section class="set">
  <div class="set-header">
    <div>
      <h2>${escapeHtml(set.dashboardName)}</h2>
      <p class="muted">${escapeHtml(set.projectId)} · ${escapeHtml(set.objective)}</p>
      ${set.selectedVariantId ? `<p class="selected-line"><strong>Selected Direction:</strong> ${escapeHtml(selectedVariantName(set))} · ${escapeHtml(set.selectionRationale || "No rationale recorded.")}</p>` : ""}
      <div class="question-list">${(set.operatorQuestions ?? []).map((question) => `<span class="chip">${escapeHtml(question)}</span>`).join("")}</div>
    </div>
    <div class="meta">${escapeHtml((set.variants ?? []).length)} variants</div>
  </div>
  <div class="variants">${(set.variants ?? []).map(renderVariant).join("\n")}</div>
</section>`;
}

function renderVariant(variant) {
  return `<article class="variant">
  <div class="variant-top">
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:start">
      <h3>${escapeHtml(variant.name)}</h3>
      <span class="status">${escapeHtml(variant.status)}</span>
    </div>
    <div class="workspace-row">${(variant.workspaceFocus ?? []).map((workspace) => `<span class="workspace">${escapeHtml(workspace)}</span>`).join("")}</div>
  </div>
  <div class="variant-body">
    <div class="mini-section">
      <div class="mini-title">Operator Workflow</div>
      <p class="muted">${escapeHtml(variant.operatorWorkflow)}</p>
    </div>
    <div class="mini-section">
      <div class="mini-title">Reference Notes</div>
      <ul>${(variant.referenceNotes ?? []).map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>
    </div>
    <div class="mini-section">
      <div class="mini-title">Data Requirements</div>
      ${(variant.dataRequirements ?? []).map(renderRequirement).join("")}
    </div>
    ${variant.previewEvidence?.length ? `<div class="mini-section"><div class="mini-title">Preview Evidence</div><ul>${variant.previewEvidence.map((evidence) => `<li>${escapeHtml(evidence.label)} · ${escapeHtml(evidence.kind)} · ${escapeHtml(evidence.path)}</li>`).join("")}</ul></div>` : ""}
    ${variant.promotedComponents?.length ? `<div class="mini-section"><div class="mini-title">Promotion Targets</div><div class="question-list">${variant.promotedComponents.map((component) => `<span class="chip">${escapeHtml(component)}</span>`).join("")}</div></div>` : ""}
  </div>
</article>`;
}

function selectedVariantName(set) {
  return set.variants?.find((variant) => variant.id === set.selectedVariantId)?.name || set.selectedVariantId;
}

function renderRequirement(requirement) {
  return `<div class="requirement">
  <span>${escapeHtml(requirement.label)}${requirement.required ? "" : " (optional)"}</span>
  <strong class="${escapeHtml(requirement.currentState)}">${escapeHtml(requirement.currentState)}</strong>
</div>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderWebData(registry) {
  return `import type { DashboardPrototypeSet } from "@hermes/dashboard-kit";

export const dashboardPrototypeSets = ${JSON.stringify(registry.prototypeSets ?? [], null, 2)} satisfies DashboardPrototypeSet[];
`;
}

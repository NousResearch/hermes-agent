#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const registryPath = path.join(root, "docs/design/dashboard-kit-adoption.json");
const prototypeRegistryPath = path.join(root, "docs/design/dashboard-prototype-registry.json");
const prototypeGalleryPath = path.join(root, "docs/design/prototype-gallery/index.html");
const downstreamFeedPath = path.join(root, "docs/design/dashboard-downstream-snapshot-feed.json");
const packageNativeParityPath = path.join(root, "docs/design/package-native-parity-registry.json");
const packageNativeCutoverPath = path.join(root, "docs/design/package-native-cutover-checklist.json");
const planPath = path.join(root, "docs/design/dashboard-design-system-spine-plan.md");
const requiredDocs = [
  "docs/design/dashboard-data-contracts.md",
  "docs/design/dashboard-downstream-snapshot-feed.md",
  "docs/design/dashboard-downstream-snapshot-feed.json",
  "docs/design/dashboard-information-architecture.md",
  "docs/design/mobbin-reference-workflow.md",
  "docs/design/dashboard-design-system-spine-plan.md",
  "docs/design/dashboard-prototype-lab.md",
  "docs/design/dashboard-prototype-registry.json",
  "docs/design/package-native-cutover-checklist.md",
  "docs/design/package-native-cutover-checklist.json",
  "docs/design/package-native-parity-registry.json",
  "docs/design/prototype-gallery/index.html",
  "packages/hermes-dashboard-kit/DESIGN.md",
  "packages/hermes-dashboard-kit/README.md",
];

function issue(severity, code, message) {
  return { severity, code, message };
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function validateRegistry(registry) {
  const issues = [];
  if (registry.schemaVersion !== 1) {
    issues.push(issue("error", "registry.schemaVersion", "schemaVersion must be 1."));
  }
  if (registry.source?.package !== "@hermes/dashboard-kit") {
    issues.push(issue("error", "registry.source.package", "source.package must be @hermes/dashboard-kit."));
  }
  if (!Array.isArray(registry.dashboards) || registry.dashboards.length === 0) {
    issues.push(issue("error", "registry.dashboards", "dashboards must include at least one dashboard."));
  }

  for (const [index, dashboard] of (registry.dashboards ?? []).entries()) {
    const prefix = `dashboards[${index}] ${dashboard.project ?? "(missing project)"}`;
    for (const field of ["project", "name", "type", "status", "targetState"]) {
      if (!dashboard[field]) issues.push(issue("error", `${prefix}.${field}`, `${field} is required.`));
    }
    for (const field of ["contractCoveragePercent", "workspaceCoveragePercent"]) {
      if (dashboard[field] === undefined) {
        issues.push(issue("warning", `${prefix}.${field}`, `${field} should be tracked.`));
      } else if (!Number.isFinite(dashboard[field]) || dashboard[field] < 0 || dashboard[field] > 100) {
        issues.push(issue("error", `${prefix}.${field}`, `${field} must be between 0 and 100.`));
      }
    }
    if (dashboard.status === "synced" && dashboard.type === "static-adapter" && !dashboard.targetPath) {
      issues.push(issue("error", `${prefix}.targetPath`, "synced static adapters must include targetPath."));
    }
    if ((dashboard.contractCoveragePercent ?? 0) >= 50 && !dashboard.architectureEndpoint) {
      issues.push(issue("warning", `${prefix}.architectureEndpoint`, "dashboards with meaningful contract coverage should expose an architecture endpoint."));
    }
  }

  return issues;
}

function validateDocs() {
  const issues = [];
  for (const relativePath of requiredDocs) {
    const absolutePath = path.join(root, relativePath);
    if (!fs.existsSync(absolutePath)) {
      issues.push(issue("error", `doc.missing:${relativePath}`, `${relativePath} is missing.`));
    }
  }
  const plan = fs.existsSync(planPath) ? fs.readFileSync(planPath, "utf8") : "";
  for (const section of ["V1 - Shared Dashboard Language", "V6 - Prototype Lab", "V7 - Enforcement", "Boundary"]) {
    if (!plan.includes(section)) {
      issues.push(issue("error", `plan.section:${section}`, `Plan is missing section: ${section}.`));
    }
  }
  return issues;
}

function validatePrototypeRegistry(registry) {
  const issues = [];
  if (registry.schemaVersion !== 1) {
    issues.push(issue("error", "prototype.schemaVersion", "prototype registry schemaVersion must be 1."));
  }
  if (!Array.isArray(registry.prototypeSets) || registry.prototypeSets.length === 0) {
    issues.push(issue("error", "prototype.sets", "prototype registry must include at least one prototype set."));
  }
  for (const [setIndex, set] of (registry.prototypeSets ?? []).entries()) {
    const setPrefix = `prototypeSets[${setIndex}] ${set.projectId ?? "(missing project)"}`;
    for (const field of ["id", "projectId", "dashboardName", "createdAt", "objective"]) {
      if (!set[field]) issues.push(issue("error", `${setPrefix}.${field}`, `${field} is required.`));
    }
    if (!Array.isArray(set.operatorQuestions) || set.operatorQuestions.length === 0) {
      issues.push(issue("error", `${setPrefix}.operatorQuestions`, "operatorQuestions are required."));
    }
    if (!Array.isArray(set.variants) || set.variants.length < 3) {
      issues.push(issue("error", `${setPrefix}.variants`, "each prototype set must include at least three variants."));
    }
    if (set.selectedVariantId && !(set.variants ?? []).some((variant) => variant.id === set.selectedVariantId)) {
      issues.push(issue("error", `${setPrefix}.selectedVariantId`, "selectedVariantId must match a variant id."));
    }
    if (set.selectedVariantId && !set.selectionRationale) {
      issues.push(issue("error", `${setPrefix}.selectionRationale`, "selected prototype sets must include a selection rationale."));
    }
    if (set.selectedVariantId && (!Array.isArray(set.selectionEvidence) || set.selectionEvidence.length === 0)) {
      issues.push(issue("error", `${setPrefix}.selectionEvidence`, "selected prototype sets must include selection evidence."));
    }
    for (const [variantIndex, variant] of (set.variants ?? []).entries()) {
      const variantPrefix = `${setPrefix}.variants[${variantIndex}] ${variant.id ?? "(missing variant)"}`;
      for (const field of ["id", "name", "status", "operatorWorkflow"]) {
        if (!variant[field]) issues.push(issue("error", `${variantPrefix}.${field}`, `${field} is required.`));
      }
      if (!Array.isArray(variant.workspaceFocus) || variant.workspaceFocus.length === 0) {
        issues.push(issue("error", `${variantPrefix}.workspaceFocus`, "workspaceFocus is required."));
      }
      if (!Array.isArray(variant.dataRequirements) || variant.dataRequirements.length === 0) {
        issues.push(issue("warning", `${variantPrefix}.dataRequirements`, "variant should list data requirements."));
      }
      if (!Array.isArray(variant.referenceNotes) || variant.referenceNotes.length === 0) {
        issues.push(issue("warning", `${variantPrefix}.referenceNotes`, "variant should list reference notes."));
      }
      if (set.selectedVariantId === variant.id && (!Array.isArray(variant.promotedComponents) || variant.promotedComponents.length === 0)) {
        issues.push(issue("error", `${variantPrefix}.promotedComponents`, "selected variants must list promotion target components."));
      }
      if (set.selectedVariantId === variant.id && (!Array.isArray(variant.previewEvidence) || variant.previewEvidence.length === 0)) {
        issues.push(issue("error", `${variantPrefix}.previewEvidence`, "selected variants must include preview evidence."));
      }
    }
  }
  return issues;
}

function validatePrototypeGallery(registry) {
  const issues = [];
  if (!fs.existsSync(prototypeGalleryPath)) {
    issues.push(issue("error", "prototype.gallery.missing", "prototype gallery has not been generated."));
    return issues;
  }
  const gallery = fs.readFileSync(prototypeGalleryPath, "utf8");
  for (const set of registry.prototypeSets ?? []) {
    if (!gallery.includes(set.dashboardName)) {
      issues.push(issue("error", `prototype.gallery.set:${set.id}`, `prototype gallery is missing ${set.dashboardName}.`));
    }
    if (set.selectedVariantId && !gallery.includes("Selected Direction")) {
      issues.push(issue("error", `prototype.gallery.selection:${set.id}`, "prototype gallery is missing selected direction content."));
    }
    for (const variant of set.variants ?? []) {
      if (!gallery.includes(variant.name)) {
        issues.push(issue("error", `prototype.gallery.variant:${set.id}.${variant.id}`, `prototype gallery is missing ${variant.name}.`));
      }
    }
  }
  return issues;
}

function validateDownstreamFeed(feed) {
  const issues = [];
  if (feed.schemaVersion !== 1) {
    issues.push(issue("error", "downstream.schemaVersion", "downstream feed schemaVersion must be 1."));
  }
  if (feed.contract?.package !== "@hermes/dashboard-kit") {
    issues.push(issue("error", "downstream.contract.package", "downstream feed package must be @hermes/dashboard-kit."));
  }
  if (!Array.isArray(feed.producers) || feed.producers.length === 0) {
    issues.push(issue("error", "downstream.producers", "downstream feed must include producers."));
  }
  if (!Array.isArray(feed.consumers) || feed.consumers.length === 0) {
    issues.push(issue("error", "downstream.consumers", "downstream feed must include consumers."));
  }

  const producerIds = new Set();
  for (const [index, producer] of (feed.producers ?? []).entries()) {
    const prefix = `downstream.producers[${index}] ${producer.projectId ?? "(missing projectId)"}`;
    for (const field of ["projectId", "label", "owner", "status", "snapshotEndpoint", "architectureEndpoint"]) {
      if (!producer[field]) issues.push(issue("error", `${prefix}.${field}`, `${field} is required.`));
    }
    if (producer.projectId) producerIds.add(producer.projectId);
    if (!Array.isArray(producer.signals) || producer.signals.length === 0) {
      issues.push(issue("error", `${prefix}.signals`, "producer signals are required."));
    }
    if (!Array.isArray(producer.consumers) || producer.consumers.length === 0) {
      issues.push(issue("error", `${prefix}.consumers`, "producer consumers are required."));
    }
  }

  for (const [index, consumer] of (feed.consumers ?? []).entries()) {
    const prefix = `downstream.consumers[${index}] ${consumer.projectId ?? "(missing projectId)"}`;
    for (const field of ["projectId", "label", "status", "purpose"]) {
      if (!consumer[field]) issues.push(issue("error", `${prefix}.${field}`, `${field} is required.`));
    }
    if (!Array.isArray(consumer.consumes) || consumer.consumes.length === 0) {
      issues.push(issue("error", `${prefix}.consumes`, "consumer consumes list is required."));
    }
    for (const producerId of consumer.consumes ?? []) {
      if (!producerIds.has(producerId)) {
        issues.push(issue("error", `${prefix}.consumes:${producerId}`, "consumer references an unknown producer."));
      }
    }
  }

  return issues;
}

function validatePackageNativeParity(registry, checklist) {
  const issues = [];
  if (registry.schemaVersion !== 1) {
    issues.push(issue("error", "packageNative.schemaVersion", "package-native parity registry schemaVersion must be 1."));
  }
  if (!Array.isArray(registry.targets) || registry.targets.length === 0) {
    issues.push(issue("error", "packageNative.targets", "package-native parity registry must include targets."));
  }
  if (checklist.schemaVersion !== 1) {
    issues.push(issue("error", "packageNativeCutover.schemaVersion", "package-native cutover checklist schemaVersion must be 1."));
  }
  if (!Array.isArray(checklist.requiredEvidence) || checklist.requiredEvidence.length === 0) {
    issues.push(issue("error", "packageNativeCutover.requiredEvidence", "cutover checklist must define required evidence."));
  }
  const checklistTargetIds = new Set((checklist.targets ?? []).map((target) => target.id));
  for (const [index, target] of (registry.targets ?? []).entries()) {
    const prefix = `packageNative.targets[${index}] ${target.id ?? "(missing id)"}`;
    for (const field of ["id", "dashboard", "currentSurface", "targetSurface", "recipe", "adapterPath", "parity", "nextStep"]) {
      if (!target[field]) issues.push(issue("error", `${prefix}.${field}`, `${field} is required.`));
    }
    if (target.retirementAllowed) {
      for (const [key, value] of Object.entries(target.parity ?? {})) {
        if (value !== true) {
          issues.push(issue("error", `${prefix}.retirementAllowed`, `retirementAllowed requires parity.${key}=true.`));
        }
      }
    }
    if (["media-engine.ops", "khashi-vc.roc"].includes(target.id) && !checklistTargetIds.has(target.id)) {
      issues.push(issue("error", `${prefix}.cutoverChecklist`, "priority package-native target must be listed in the cutover checklist."));
    }
  }
  return issues;
}

const registry = readJson(registryPath);
const prototypeRegistry = readJson(prototypeRegistryPath);
const downstreamFeed = readJson(downstreamFeedPath);
const packageNativeParity = readJson(packageNativeParityPath);
const packageNativeCutover = readJson(packageNativeCutoverPath);
const issues = [
  ...validateRegistry(registry),
  ...validatePrototypeRegistry(prototypeRegistry),
  ...validatePrototypeGallery(prototypeRegistry),
  ...validateDownstreamFeed(downstreamFeed),
  ...validatePackageNativeParity(packageNativeParity, packageNativeCutover),
  ...validateDocs(),
];
const errors = issues.filter((item) => item.severity === "error");
const warnings = issues.filter((item) => item.severity === "warning");

for (const item of issues) {
  console.log(`${item.severity.toUpperCase()} ${item.code}: ${item.message}`);
}

console.log(`Dashboard spine validation: ${errors.length} error(s), ${warnings.length} warning(s).`);
if (errors.length) process.exit(1);

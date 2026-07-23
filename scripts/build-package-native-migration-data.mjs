#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const registryPath = path.join(root, "docs/design/package-native-parity-registry.json");
const webDataPath = path.join(root, "web/src/pages/package-native-migration-data.ts");

const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"));
fs.writeFileSync(webDataPath, renderWebData(registry));
console.log(`Wrote ${path.relative(root, webDataPath)}`);

function renderWebData(registry) {
  const targets = (registry.targets ?? []).map((target) => {
    const parityEntries = Object.entries(target.parity ?? {});
    const passed = parityEntries.filter(([, value]) => value === true).length;
    const completion = parityEntries.length
      ? Math.round((passed / parityEntries.length) * 100)
      : 0;
    return {
      id: target.id,
      dashboard: target.dashboard,
      recipe: target.recipe,
      currentSurface: target.currentSurface,
      targetSurface: target.targetSurface,
      packageNativeRoute: target.packageNativeRoute ?? null,
      snapshotEndpoint: target.snapshotEndpoint ?? null,
      healthEndpoint: target.healthEndpoint ?? null,
      adapterPath: target.adapterPath,
      completion,
      status: migrationStatus(target),
      nextStep: target.nextStep,
      retirementAllowed: Boolean(target.retirementAllowed),
      parity: target.parity ?? {},
    };
  });

  return `export interface PackageNativeMigrationTarget {
  id: string;
  dashboard: string;
  recipe: string;
  currentSurface: string;
  targetSurface: string;
  packageNativeRoute: string | null;
  snapshotEndpoint: string | null;
  healthEndpoint: string | null;
  adapterPath: string;
  completion: number;
  status: "ready" | "in-progress" | "blocked" | "planned";
  nextStep: string;
  retirementAllowed: boolean;
  parity: Record<string, boolean>;
}

export const packageNativeMigrationTargets: PackageNativeMigrationTarget[] = ${JSON.stringify(targets, null, 2)};
`;
}

function migrationStatus(target) {
  if (target.retirementAllowed) return "ready";
  if (!target.packageNativeRoute) return "planned";
  if (target.parity?.productionScreenshotEvidence === false) return "blocked";
  return "in-progress";
}

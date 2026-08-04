/**
 * Manifest — self-describing metadata for a memory-tdai data directory.
 *
 * Lives at `<dataDir>/.metadata/manifest.json`.
 *
 * - **store**: written once on first successful store init; never overwritten.
 *   On subsequent starts the current config is compared against the persisted
 *   store binding — mismatches are logged at debug level (informational only).
 * - **seed**: written once when a seed run completes; null for live-runtime dirs.
 *
 * This file is informational / read-only from the user's perspective.
 * The plugin reads it on startup for consistency checks.
 */

import fs from "node:fs";
import path from "node:path";

// ============================
// Types
// ============================

export interface ManifestStoreInfo {
  type: "sqlite" | "tcvdb";
  sqlite?: {
    /** Relative path to the SQLite DB file (relative to dataDir). */
    path: string;
  };
  tcvdb?: {
    url: string;
    database: string;
    /** User-friendly alias (optional). */
    alias?: string;
  };
}

export interface ManifestSeedInfo {
  /** Original input file name (basename only). */
  inputFile?: string;
  sessions: number;
  rounds: number;
  messages: number;
  startedAt: string;
  completedAt: string;
}

export interface Manifest {
  /** Schema version for future migrations. */
  version: 1;
  /** Timestamp when the manifest was first created. */
  createdAt: string;
  /** Store binding — written once on first init. */
  store: ManifestStoreInfo;
  /** Seed run info — null for live-runtime directories. */
  seed: ManifestSeedInfo | null;
}

// ============================
// Paths
// ============================

const METADATA_DIR = ".metadata";
const MANIFEST_FILE = "manifest.json";

export function manifestPath(dataDir: string): string {
  return path.join(dataDir, METADATA_DIR, MANIFEST_FILE);
}

// ============================
// Read / Write
// ============================

/**
 * Read an existing manifest from disk. Returns `null` if not found or unparseable.
 */
export function readManifest(dataDir: string): Manifest | null {
  const p = manifestPath(dataDir);
  try {
    if (!fs.existsSync(p)) return null;
    const raw = fs.readFileSync(p, "utf-8");
    return JSON.parse(raw) as Manifest;
  } catch {
    return null;
  }
}

/**
 * Write a manifest to disk (creates `.metadata/` if needed).
 */
export function writeManifest(dataDir: string, manifest: Manifest): void {
  const dir = path.join(dataDir, METADATA_DIR);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    manifestPath(dataDir),
    JSON.stringify(manifest, null, 2) + "\n",
    "utf-8",
  );
}

// ============================
// Store binding helpers
// ============================

export interface StoreConfigSnapshot {
  type: "sqlite" | "tcvdb";
  sqlitePath?: string;
  tcvdbUrl?: string;
  tcvdbDatabase?: string;
  tcvdbAlias?: string;
}

/**
 * Build a ManifestStoreInfo from the current store config snapshot.
 */
export function buildStoreInfo(snapshot: StoreConfigSnapshot): ManifestStoreInfo {
  const info: ManifestStoreInfo = { type: snapshot.type };
  if (snapshot.type === "sqlite") {
    info.sqlite = { path: snapshot.sqlitePath ?? "vectors.db" };
  } else {
    info.tcvdb = {
      url: snapshot.tcvdbUrl!,
      database: snapshot.tcvdbDatabase!,
      alias: snapshot.tcvdbAlias || undefined,
    };
  }
  return info;
}

/**
 * Compare the persisted store binding against the current config.
 * Returns a list of human-readable mismatch descriptions (empty = all good).
 */
export function diffStoreBinding(
  persisted: ManifestStoreInfo,
  current: ManifestStoreInfo,
): string[] {
  const diffs: string[] = [];

  if (persisted.type !== current.type) {
    diffs.push(`store type changed: ${persisted.type} → ${current.type}`);
    return diffs; // no point comparing fields across different types
  }

  if (persisted.type === "sqlite" && current.type === "sqlite") {
    if (persisted.sqlite?.path !== current.sqlite?.path) {
      diffs.push(`sqlite path changed: ${persisted.sqlite?.path} → ${current.sqlite?.path}`);
    }
  }

  if (persisted.type === "tcvdb" && current.type === "tcvdb") {
    if (persisted.tcvdb?.url !== current.tcvdb?.url) {
      diffs.push(`tcvdb url changed: ${persisted.tcvdb?.url} → ${current.tcvdb?.url}`);
    }
    if (persisted.tcvdb?.database !== current.tcvdb?.database) {
      diffs.push(`tcvdb database changed: ${persisted.tcvdb?.database} → ${current.tcvdb?.database}`);
    }
  }

  return diffs;
}

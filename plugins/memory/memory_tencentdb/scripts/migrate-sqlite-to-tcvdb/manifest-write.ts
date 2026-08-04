import fs from "node:fs/promises";
import path from "node:path";
import {
  buildStoreInfo,
  manifestPath,
  readManifest,
  writeManifest,
} from "../../src/utils/manifest.js";

export interface RewriteMigrationManifestParams {
  dataDir: string;
  tcvdbUrl: string;
  tcvdbDatabase: string;
  tcvdbAlias?: string;
  backupExisting?: boolean;
  now?: () => string;
}

export interface RewriteMigrationManifestResult {
  created: boolean;
  updated: boolean;
  backupPath?: string;
}

const DEFAULT_BACKUP_FILE = "manifest.json.migrate.bak";

export async function rewriteMigrationManifest(
  params: RewriteMigrationManifestParams,
): Promise<RewriteMigrationManifestResult> {
  const existing = readManifest(params.dataDir);
  const nextStore = buildStoreInfo({
    type: "tcvdb",
    tcvdbUrl: params.tcvdbUrl,
    tcvdbDatabase: params.tcvdbDatabase,
    tcvdbAlias: params.tcvdbAlias,
  });

  if (!existing) {
    writeManifest(params.dataDir, {
      version: 1,
      createdAt: params.now?.() ?? new Date().toISOString(),
      store: nextStore,
      seed: null,
    });
    return {
      created: true,
      updated: false,
      backupPath: undefined,
    };
  }

  let backupPath: string | undefined;
  if (params.backupExisting !== false) {
    backupPath = path.join(path.dirname(manifestPath(params.dataDir)), DEFAULT_BACKUP_FILE);
    await fs.mkdir(path.dirname(backupPath), { recursive: true });
    await fs.copyFile(manifestPath(params.dataDir), backupPath);
  }

  writeManifest(params.dataDir, {
    ...existing,
    store: nextStore,
  });

  return {
    created: false,
    updated: true,
    backupPath,
  };
}

import fs from "node:fs/promises";
import JSON5 from "json5";

export type Bm25Language = "zh" | "en";

export interface MigrationPluginConfigTarget {
  url: string;
  username: string;
  apiKey: string;
  database: string;
  alias: string;
  embeddingModel: string;
  timeout: number;
  bm25Enabled: boolean;
  bm25Language: Bm25Language;
}

export interface WriteMigrationPluginConfigParams {
  configPath: string;
  pluginId: string;
  tcvdb: {
    url: string;
    username: string;
    apiKey: string;
    database: string;
    alias: string;
    embeddingModel: string;
    timeout: number;
  };
  bm25: {
    enabled: boolean;
    language: Bm25Language;
  };
}

interface ConfigWriteAdapterDeps {
  fs?: Pick<typeof fs, "readFile" | "writeFile" | "mkdir">;
  parseConfig?: (raw: string) => unknown;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? { ...(value as Record<string, unknown>) }
    : {};
}

export function buildMigrationPluginConfigPatch(target: MigrationPluginConfigTarget): Record<string, unknown> {
  return {
    storeBackend: "tcvdb",
    tcvdb: {
      url: target.url,
      username: target.username,
      apiKey: target.apiKey,
      database: target.database,
      alias: target.alias,
      embeddingModel: target.embeddingModel,
      timeout: target.timeout,
    },
    bm25: {
      enabled: target.bm25Enabled,
      language: target.bm25Language,
    },
  };
}

function applyPluginConfigPatch(
  sourceConfig: Record<string, unknown>,
  pluginId: string,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  const nextConfig = asRecord(sourceConfig);
  const plugins = asRecord(nextConfig.plugins);
  const entries = asRecord(plugins.entries);
  const targetEntry = asRecord(entries[pluginId]);
  const targetPluginConfig = asRecord(targetEntry.config);
  const patchTcvdb = asRecord(patch.tcvdb);
  const patchBm25 = asRecord(patch.bm25);

  const mergedPluginConfig: Record<string, unknown> = {
    ...targetPluginConfig,
    ...patch,
    tcvdb: {
      ...asRecord(targetPluginConfig.tcvdb),
      ...patchTcvdb,
    },
    bm25: {
      ...asRecord(targetPluginConfig.bm25),
      ...patchBm25,
    },
  };

  entries[pluginId] = {
    ...targetEntry,
    config: mergedPluginConfig,
  };

  plugins.entries = entries;
  nextConfig.plugins = plugins;
  return nextConfig;
}

export async function writeMigrationPluginConfig(
  params: WriteMigrationPluginConfigParams,
  deps: ConfigWriteAdapterDeps = {},
): Promise<void> {
  const fsImpl = deps.fs ?? fs;
  const parseConfig = deps.parseConfig ?? ((raw: string) => JSON5.parse(raw));

  let parsed: unknown;
  try {
    parsed = parseConfig(await fsImpl.readFile(params.configPath, "utf-8"));
  } catch {
    throw new Error(
      `Config migration writer only supports single-file JSON/JSON5: ${params.configPath}`,
    );
  }

  const nextConfig = applyPluginConfigPatch(
    asRecord(parsed),
    params.pluginId,
    buildMigrationPluginConfigPatch({
      ...params.tcvdb,
      bm25Enabled: params.bm25.enabled,
      bm25Language: params.bm25.language,
    }),
  );

  await fsImpl.writeFile(params.configPath, `${JSON.stringify(nextConfig, null, 2)}\n`, "utf-8");
}

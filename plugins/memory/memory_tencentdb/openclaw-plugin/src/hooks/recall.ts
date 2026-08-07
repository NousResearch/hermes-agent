/**
 * Recall hook (v3): search memories from Gateway + format prompt injection.
 */

import type { MemoryClient } from "@tencentdb-agent-memory/memory-sdk-ts-v2";
import { formatRecallResult } from "../format.js";

const TAG = "[memory-client-v3][recall]";

interface Logger {
  debug?: (msg: string) => void;
  info: (msg: string) => void;
  warn: (msg: string) => void;
  error: (msg: string) => void;
}

export interface RecallOptions {
  query: string;
  maxResults: number;
  includePersona: boolean;
  includeSceneNav: boolean;
}

export interface RecallResult {
  prependContext?: string;
  appendSystemContext?: string;
}

export async function performRecall(
  client: MemoryClient,
  opts: RecallOptions,
  logger?: Logger,
): Promise<RecallResult> {
  const startMs = Date.now();

  // Parallel requests: L1 search + L3 persona + L2 scenario list
  const [searchResult, persona, scenarios] = await Promise.allSettled([
    client.searchAtomic({ query: opts.query, limit: opts.maxResults }),
    opts.includePersona ? client.readCore() : Promise.resolve(null),
    opts.includeSceneNav ? client.listScenarios({}) : Promise.resolve(null),
  ]);

  // Extract results (graceful on failures)
  const l1Items = searchResult.status === "fulfilled" ? (searchResult.value?.items ?? []) : [];
  const personaContent = persona.status === "fulfilled" && persona.value ? persona.value.content : null;
  const sceneEntries = scenarios.status === "fulfilled" && scenarios.value ? (scenarios.value.entries ?? []) : [];

  const elapsedMs = Date.now() - startMs;
  logger?.info(
    `${TAG} Recall complete (${elapsedMs}ms): L1=${l1Items.length}, ` +
    `persona=${personaContent ? "yes" : "no"}, scenes=${sceneEntries.length}`,
  );

  return formatRecallResult(l1Items, personaContent, sceneEntries);
}

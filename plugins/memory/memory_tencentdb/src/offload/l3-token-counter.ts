/**
 * L3 token counting: prefer tiktoken (exact for OpenAI-style BPE), with heuristic fallback.
 */
import { getEncoding, type Tiktoken } from "js-tiktoken";
import { PLUGIN_DEFAULTS, type PluginConfig, type PluginLogger } from "./types.js";
import { estimateL3MixedTokensHeuristic } from "./l3-token-helpers.js";

export function createL3TokenCounter(
  pluginConfig: Partial<PluginConfig> | undefined,
  logger: PluginLogger | undefined,
): (text: string) => number {
  const mode =
    (pluginConfig as any)?.l3TokenCountMode ?? PLUGIN_DEFAULTS.l3TokenCountMode;
  if (mode === "heuristic") {
    return (text: string) => estimateL3MixedTokensHeuristic(text);
  }
  const encodingName: string =
    ((pluginConfig as any)?.l3TiktokenEncoding ??
      PLUGIN_DEFAULTS.l3TiktokenEncoding) as string;
  let enc: Tiktoken | null = null;
  return (text: string): number => {
    try {
      if (!enc) {
        enc = getEncoding(encodingName as any);
        logger?.debug?.(`[context-offload] L3 token counter: tiktoken encoding=${encodingName}`);
      }
      return enc!.encode(text).length;
    } catch (err) {
      logger?.warn?.(
        `[context-offload] tiktoken encode failed (${String(err)}), falling back to heuristic`,
      );
      return estimateL3MixedTokensHeuristic(text);
    }
  };
}

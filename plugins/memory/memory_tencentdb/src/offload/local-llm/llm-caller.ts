/**
 * Unified LLM caller for offload local mode.
 *
 * Uses Vercel AI SDK (`ai` + `@ai-sdk/openai`) with "compatible" mode
 * to support any OpenAI-compatible backend.
 */
import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import type { PluginLogger } from "../types.js";

const TAG = "[context-offload] [local-llm]";

export interface LlmCallerConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  temperature: number;
  timeoutMs: number;
}

export interface CallLlmOpts {
  systemPrompt: string;
  userPrompt: string;
  /** Override temperature for this call */
  temperature?: number;
  /** Override timeout for this call */
  timeoutMs?: number;
  /** Label for logging (e.g. "L1", "L1.5", "L2") */
  label?: string;
  /** Instance ID for telemetry metadata */
  instanceId?: string;
}

/**
 * Call LLM with the given prompts and return the text response.
 * Throws on timeout or API errors.
 */
export async function callLlm(
  config: LlmCallerConfig,
  opts: CallLlmOpts,
  logger?: PluginLogger,
): Promise<string> {
  const startMs = Date.now();
  const label = opts.label ?? "call";
  const temperature = opts.temperature ?? config.temperature;
  const timeoutMs = opts.timeoutMs ?? config.timeoutMs;

  logger?.info?.(
    `${TAG} ${label} >>> model=${config.model}, temp=${temperature}, timeout=${timeoutMs}ms, ` +
    `systemLen=${opts.systemPrompt.length}, userLen=${opts.userPrompt.length}`,
  );

  const provider = createOpenAI({
    baseURL: config.baseUrl,
    apiKey: config.apiKey,
    compatibility: "compatible",
  });

  try {
    const result = await generateText({
      model: provider.chat(config.model),
      system: opts.systemPrompt,
      prompt: opts.userPrompt,
      temperature,
      abortSignal: AbortSignal.timeout(timeoutMs),
      experimental_telemetry: {
        isEnabled: true,
        functionId: opts.label ?? "offload-llm",
        metadata: { instanceId: opts.instanceId ?? "unknown" },
      },
    });

    const text = result.text.trim();
    const elapsedMs = Date.now() - startMs;

    logger?.info?.(
      `${TAG} ${label} <<< ${elapsedMs}ms, output=${text.length} chars`,
    );

    return text;
  } catch (err) {
    const elapsedMs = Date.now() - startMs;
    const errMsg = err instanceof Error ? err.message : String(err);
    logger?.error?.(`${TAG} ${label} FAILED (${elapsedMs}ms): ${errMsg}`);
    throw err;
  }
}

/**
 * Seed runtime: L0→L1→L2→L3 orchestration for the `seed` command.
 *
 * Uses the shared pipeline-factory for VectorStore/EmbeddingService init,
 * L1 runner, L2 runner, L3 runner, and persister wiring — keeping this
 * module focused on seed-specific concerns:
 * - Synchronous per-round L0 capture with progress reporting
 * - waitForL1Idle polling (L1 only — see FIXME below)
 * - Ctrl+C graceful shutdown
 *
 * FIXME: Currently we only wait for L1 to become idle before destroying the
 * pipeline.  L2 (scene extraction) and L3 (persona generation) may still be
 * in-flight when `pipeline.destroy()` is called.  This is intentional for now
 * to avoid excessively long seed runs, but means seed output may not include
 * the latest L2/L3 artifacts.  Re-evaluate adding a full L1+L2+L3 idle wait
 * once pipeline-manager exposes reliable L2/L3 idle signals.
 */

import path from "node:path";
import { parseConfig } from "../../config.js";
import type { MemoryTdaiConfig } from "../../config.js";
import { performAutoCapture } from "../hooks/auto-capture.js";
import { createPipeline, createL2Runner, createL3Runner } from "../../utils/pipeline-factory.js";
import type { PipelineInstance, PipelineLogger } from "../../utils/pipeline-factory.js";
import { readManifest, writeManifest } from "../../utils/manifest.js";
import { StandaloneLLMRunnerFactory } from "../../adapters/standalone/llm-runner.js";
import type { MemoryPipelineManager } from "../../utils/pipeline-manager.js";
import type { LLMRunner } from "../types.js";
import type {
  NormalizedInput,
  SeedProgress,
  SeedSummary,
} from "./types.js";

const TAG = "[memory-tdai] [seed]";

// ============================
// Seed pipeline options
// ============================

export interface SeedRuntimeOptions {
  /** Directory to store all seed output (L0, checkpoint, vectors.db). */
  outputDir: string;
  /** OpenClaw config object (needed for LLM calls in L1). */
  openclawConfig: unknown;
  /** Raw plugin config (same shape as api.pluginConfig). */
  pluginConfig?: Record<string, unknown>;
  /** Original input file path (for manifest traceability). */
  inputFile?: string;
  /** Logger instance. */
  logger: PipelineLogger;
  /** Progress callback (called after each round). */
  onProgress?: (progress: SeedProgress) => void;
}

// ============================
// Seed pipeline creation
// ============================

/**
 * Create a seed pipeline using the shared factory, with L2/L3 runners
 * wired via shared factory functions (same logic as index.ts live runtime).
 */
async function createSeedPipeline(opts: SeedRuntimeOptions): Promise<{ pipeline: PipelineInstance; cfg: MemoryTdaiConfig }> {
  const { outputDir, openclawConfig, pluginConfig, logger } = opts;

  // Parse config — all values come from pluginConfig (or parseConfig defaults)
  const cfg = parseConfig(pluginConfig);

  logger.info(
    `${TAG} Creating seed pipeline: outputDir=${outputDir}, ` +
    `everyN=${cfg.pipeline.everyNConversations}, l1Idle=${cfg.pipeline.l1IdleTimeoutSeconds}s, ` +
    `l2Delay=${cfg.pipeline.l2DelayAfterL1Seconds}s, l2Min=${cfg.pipeline.l2MinIntervalSeconds}s, l2Max=${cfg.pipeline.l2MaxIntervalSeconds}s`,
  );

  // Create standalone LLM runners if cfg.llm is configured.
  // Seed always runs outside OpenClaw, so it needs standalone runners
  // unless an explicit openclawConfig is provided (rare).
  let l1LlmRunner: LLMRunner | undefined;
  let l2l3LlmRunner: LLMRunner | undefined;

  if (cfg.llm.enabled && cfg.llm.apiKey) {
    const runnerFactory = new StandaloneLLMRunnerFactory({
      config: {
        baseUrl: cfg.llm.baseUrl,
        apiKey: cfg.llm.apiKey,
        model: cfg.llm.model,
        maxTokens: cfg.llm.maxTokens,
        timeoutMs: cfg.llm.timeoutMs,
      },
      logger,
    });
    l1LlmRunner = runnerFactory.createRunner({ enableTools: false });
    l2l3LlmRunner = runnerFactory.createRunner({ enableTools: true });
    logger.info(`${TAG} Seed using standalone LLM: model=${cfg.llm.model}`);
  }

  // Use shared factory for everything: store init, L1 runner, persister, destroy
  const pipeline = await createPipeline({
    pluginDataDir: outputDir,
    cfg,
    openclawConfig,
    logger,
    l1LlmRunner,
  });

  // Wire L2 runner via shared factory (same logic as index.ts live runtime)
  pipeline.scheduler.setL2Runner(createL2Runner({
    pluginDataDir: outputDir,
    cfg,
    openclawConfig,
    vectorStore: pipeline.vectorStore,
    logger,
    llmRunner: l2l3LlmRunner,
  }));

  // Wire L3 runner via shared factory (same logic as index.ts live runtime)
  pipeline.scheduler.setL3Runner(createL3Runner({
    pluginDataDir: outputDir,
    cfg,
    openclawConfig,
    vectorStore: pipeline.vectorStore,
    logger,
    llmRunner: l2l3LlmRunner,
  }));

  return { pipeline, cfg };
}

// ============================
// waitForL1Idle
// ============================

/**
 * Poll pipeline queue status until L1 is idle for a given session.
 * Modeled after benchmark-ingest.ts waitForPipelineIdle() but focused on L1 only.
 */
async function waitForL1Idle(
  scheduler: MemoryPipelineManager,
  sessionKeys: string[],
  logger: PipelineLogger,
  opts: {
    pollIntervalMs?: number;
    stableRounds?: number;
    maxWaitMs?: number;
  } = {},
): Promise<void> {
  const pollInterval = opts.pollIntervalMs ?? 1_000;
  const stableRounds = opts.stableRounds ?? 3;
  const maxWait = opts.maxWaitMs ?? 300_000; // 5 min default

  const startTime = Date.now();
  let consecutiveIdle = 0;

  while (true) {
    const elapsed = Date.now() - startTime;
    if (elapsed > maxWait) {
      logger.warn(`${TAG} [waitL1] Max wait time reached (${(maxWait / 1000).toFixed(0)}s), proceeding`);
      break;
    }

    const queues = scheduler.getQueueSizes();

    // Check per-session: buffered messages + conversation count
    let totalBuffered = 0;
    let totalConversationCount = 0;
    for (const key of sessionKeys) {
      totalBuffered += scheduler.getBufferedMessageCount(key);
      const state = scheduler.getSessionState(key);
      if (state) {
        totalConversationCount += state.conversation_count;
      }
    }

    const isIdle =
      queues.l1Idle &&
      totalBuffered === 0 &&
      totalConversationCount === 0;

    if (isIdle) {
      consecutiveIdle++;
      if (consecutiveIdle >= stableRounds) {
        logger.debug?.(`${TAG} [waitL1] L1 stable for ${stableRounds} consecutive polls`);
        return;
      }
    } else {
      consecutiveIdle = 0;
      logger.debug?.(
        `${TAG} [waitL1] Waiting: l1Queue=${queues.l1}, l1Pending=${queues.l1Pending}, l1Idle=${queues.l1Idle}, ` +
        `buffered=${totalBuffered}, convCount=${totalConversationCount}`,
      );
    }

    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }
}

// ============================
// Main execution function
// ============================

/**
 * Execute the seed pipeline: feed normalized input through L0 → L1.
 *
 * L2/L3 runners are wired but their completion is **not** awaited — see the
 * module-level FIXME.  The pipeline is destroyed after L1 idle, so L2/L3 may
 * be interrupted mid-run.
 *
 * This is the core runtime called by `src/cli/commands/seed.ts` after
 * all input validation and user confirmation are complete.
 */
export async function executeSeed(
  input: NormalizedInput,
  opts: SeedRuntimeOptions,
): Promise<SeedSummary> {
  const { logger, onProgress } = opts;
  const startTime = Date.now();

  // Track interrupt signal
  let interrupted = false;
  const onSigint = () => {
    if (interrupted) {
      // Second Ctrl+C — force exit
      logger.warn(`${TAG} Force exit (second Ctrl+C)`);
      process.exit(1);
    }
    interrupted = true;
    logger.warn(`${TAG} Interrupt received, finishing current round and shutting down...`);
  };
  process.on("SIGINT", onSigint);

  let pipeline: PipelineInstance | undefined;
  let totalL0Recorded = 0;
  let roundsProcessed = 0;

  try {
    // Create and start pipeline (returns both the pipeline instance and the
    // seed-optimized config so we don't need to parse config again)
    const seed = await createSeedPipeline(opts);
    pipeline = seed.pipeline;
    const seedCfg = seed.cfg;

    pipeline.scheduler.start({});
    logger.info(`${TAG} Pipeline started, processing ${input.sessions.length} session(s), ${input.totalRounds} round(s)`);

    // Seed-specific: use 0 so the cold-start guard in captureAtomically()
    // does NOT filter out historical messages. In live mode Date.now()
    // prevents the first agent_end from dumping full session history,
    // but seed intentionally feeds all historical data.
    const captureStartTimestamp = 0;

    // Process each session → each round
    // Key invariant: after every everyNConversations rounds we must wait for L1
    // to finish before feeding more rounds. Without this pause the for-loop
    // would dump all rounds into L0 back-to-back and L1 would only run once
    // with the full batch (defeating the "every N" batching semantics).
    const everyN = seedCfg.pipeline.everyNConversations;

    for (const session of input.sessions) {
      if (interrupted) break;

      logger.info(`${TAG} Session: key="${session.sessionKey}" id="${session.sessionId}" rounds=${session.rounds.length}`);

      for (let ri = 0; ri < session.rounds.length; ri++) {
        if (interrupted) break;

        const round = session.rounds[ri]!;
        roundsProcessed++;

        // Build messages in the format expected by performAutoCapture.
        // Field must be named "timestamp" (not "ts") because l0-recorder's
        // extractUserAssistantMessages reads m.timestamp for incremental filtering.
        const messages = round.messages.map((m) => ({
          role: m.role,
          content: m.content,
          timestamp: m.timestamp,
        }));

        try {
          const result = await performAutoCapture({
            messages,
            sessionKey: session.sessionKey,
            sessionId: session.sessionId,
            cfg: seedCfg,
            pluginDataDir: opts.outputDir,
            logger,
            scheduler: pipeline.scheduler,
            pluginStartTimestamp: captureStartTimestamp,
            vectorStore: pipeline.vectorStore,
            embeddingService: pipeline.embeddingService,
          });

          totalL0Recorded += result.l0RecordedCount;
        } catch (err) {
          logger.error(
            `${TAG} L0 capture failed for session="${session.sessionKey}" round=${ri}: ` +
            `${err instanceof Error ? err.message : String(err)}`,
          );
        }

        // Report progress
        onProgress?.({
          currentRound: roundsProcessed,
          totalRounds: input.totalRounds,
          sessionKey: session.sessionKey,
          stage: "l0_captured",
        });

        // After every N rounds, wait for the triggered L1 to finish before
        // feeding the next batch. This keeps L1 batches aligned with the
        // everyNConversations boundary instead of letting all rounds pile up.
        const roundInSession = ri + 1; // 1-based
        if (roundInSession % everyN === 0 && !interrupted) {
          onProgress?.({
            currentRound: roundsProcessed,
            totalRounds: input.totalRounds,
            sessionKey: session.sessionKey,
            stage: "l1_waiting",
          });

          logger.info(
            `${TAG} Pausing after round ${roundInSession}/${session.rounds.length} ` +
            `for session="${session.sessionKey}" — waiting for L1 to drain`,
          );

          await waitForL1Idle(
            pipeline.scheduler,
            [session.sessionKey],
            logger,
            { pollIntervalMs: 500, stableRounds: 2, maxWaitMs: 120_000 },
          );
        }
      }

      // After all rounds for this session, wait for any residual L1 work
      // (handles the tail when total rounds is not a multiple of everyN)
      if (!interrupted) {
        onProgress?.({
          currentRound: roundsProcessed,
          totalRounds: input.totalRounds,
          sessionKey: session.sessionKey,
          stage: "l1_waiting",
        });

        await waitForL1Idle(
          pipeline.scheduler,
          [session.sessionKey],
          logger,
          { pollIntervalMs: 1_000, stableRounds: 3, maxWaitMs: 300_000 },
        );

        logger.info(`${TAG} L1 idle for session="${session.sessionKey}"`);
      }
    }

    // Final wait for all sessions
    if (!interrupted) {
      const allKeys = input.sessions.map((s) => s.sessionKey);
      logger.info(`${TAG} Final L1 idle wait for all sessions...`);
      await waitForL1Idle(
        pipeline.scheduler,
        allKeys,
        logger,
        { pollIntervalMs: 1_000, stableRounds: 3, maxWaitMs: 300_000 },
      );
    }
  } finally {
    process.removeListener("SIGINT", onSigint);

    // Graceful shutdown
    if (pipeline) {
      try {
        await pipeline.destroy();
      } catch (err) {
        logger.error(`${TAG} Pipeline destroy error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  const durationMs = Date.now() - startTime;

  const summary: SeedSummary = {
    sessionsProcessed: input.sessions.length,
    roundsProcessed,
    messagesProcessed: input.totalMessages,
    l0RecordedCount: totalL0Recorded,
    durationMs,
    outputDir: opts.outputDir,
  };

  if (interrupted) {
    logger.warn(`${TAG} Seed interrupted after ${roundsProcessed}/${input.totalRounds} rounds`);
  } else {
    logger.info(
      `${TAG} Seed complete: sessions=${summary.sessionsProcessed}, ` +
      `rounds=${summary.roundsProcessed}, messages=${summary.messagesProcessed}, ` +
      `l0Recorded=${summary.l0RecordedCount}, duration=${(durationMs / 1000).toFixed(1)}s`,
    );
  }

  // Append seed info to manifest (non-fatal if it fails)
  try {
    const manifest = readManifest(opts.outputDir);
    if (manifest) {
      manifest.seed = {
        inputFile: opts.inputFile ? path.basename(opts.inputFile) : undefined,
        sessions: summary.sessionsProcessed,
        rounds: summary.roundsProcessed,
        messages: summary.messagesProcessed,
        startedAt: new Date(startTime).toISOString(),
        completedAt: new Date().toISOString(),
      };
      writeManifest(opts.outputDir, manifest);
      logger.info(`${TAG} Manifest updated with seed info`);
    }
  } catch (err) {
    logger.warn(`${TAG} Failed to update manifest with seed info (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
  }

  return summary;
}

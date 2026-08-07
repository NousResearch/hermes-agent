/**
 * `openclaw memory-tdai seed` command definition.
 *
 * Responsibilities:
 * - Define CLI parameters and help text
 * - Interactive confirmation for timestamp auto-fill
 * - Output directory resolution and checkpoint detection
 * - Delegate to seed-runtime for actual execution
 */

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import type { Command } from "commander";
import type { SeedCliContext } from "../index.ts";
import type { SeedCommandOptions } from "../../core/seed/types.js";
import { loadAndValidateInput, fillTimestamps, SeedValidationError } from "../../core/seed/input.js";
import { executeSeed } from "../../core/seed/seed-runtime.js";

const TAG = "[memory-tdai] [seed-cmd]";

/**
 * Register the `seed` subcommand under the memory-tdai CLI namespace.
 */
export function registerSeedCommand(parent: Command, ctx: SeedCliContext): void {
  parent
    .command("seed")
    .description("Seed historical conversation data into the memory pipeline (L0 → L1)")
    .requiredOption("--input <file>", "Path to input JSON file")
    .option("--output-dir <dir>", "Output directory for pipeline data (default: auto-generated)")
    .option("--session-key <key>", "Fallback session key when input lacks one")
    .option("--config <file>", "Path to memory-tdai config override file (JSON, deep-merged on top of current plugin config)")
    .option("--strict-round-role", "Require each round to have both user and assistant messages", false)
    .option("--yes", "Skip interactive confirmations (e.g. timestamp auto-fill)", false)
    .addHelpText("after", `
Examples:
  openclaw memory-tdai seed --input conversations.json
  openclaw memory-tdai seed --input data.json --output-dir ./seed-output --strict-round-role
  openclaw memory-tdai seed --input data.json --config ./seed-config.json
  openclaw memory-tdai seed --input data.json --yes
`)
    .action(async (rawOpts: Record<string, unknown>) => {
      const opts: SeedCommandOptions = {
        input: rawOpts.input as string,
        outputDir: rawOpts.outputDir as string | undefined,
        sessionKey: rawOpts.sessionKey as string | undefined,
        strictRoundRole: rawOpts.strictRoundRole === true,
        yes: rawOpts.yes === true,
        configFile: rawOpts.config as string | undefined,
      };

      await runSeedCommand(opts, ctx);
    });
}

// ============================
// Command handler
// ============================

async function runSeedCommand(opts: SeedCommandOptions, ctx: SeedCliContext): Promise<void> {
  const { logger } = ctx;

  logger.info(`${TAG} Starting seed command...`);
  logger.info(`${TAG}   input:      ${opts.input}`);
  logger.info(`${TAG}   outputDir:  ${opts.outputDir ?? "(auto)"}`);
  logger.info(`${TAG}   sessionKey: ${opts.sessionKey ?? "(from input)"}`);
  logger.info(`${TAG}   config:     ${opts.configFile ?? "(default)"}`);
  logger.info(`${TAG}   strict:     ${opts.strictRoundRole}`);
  logger.info(`${TAG}   yes:        ${opts.yes}`);

  // 0. Load config override file and deep-merge with base plugin config
  const mergedPluginConfig = loadAndMergePluginConfig(
    ctx.pluginConfig as Record<string, unknown> | undefined,
    opts.configFile,
    logger,
  );

  // 1. Load and validate input
  let loadResult;
  try {
    loadResult = loadAndValidateInput(opts);
  } catch (err) {
    if (err instanceof SeedValidationError) {
      console.error(`\n❌ ${err.message}\n`);
      process.exit(1);
    }
    throw err;
  }

  const { input, needsTimestampConfirmation } = loadResult;

  console.log(
    `\n📥 Input loaded: ${input.sessions.length} session(s), ` +
    `${input.totalRounds} round(s), ${input.totalMessages} message(s)` +
    `${input.hasTimestamps ? "" : " (no timestamps)"}`,
  );

  // 2. Timestamp confirmation (if all messages lack timestamps)
  if (needsTimestampConfirmation) {
    if (opts.yes) {
      console.log("   Timestamps missing — auto-filling with current time (--yes)");
      fillTimestamps(input);
    } else {
      const confirmed = await askConfirmation(
        "All messages have no timestamp. Use current time for each conversation round? [y/N] ",
      );
      if (!confirmed) {
        console.log("Aborted.");
        process.exit(0);
      }
      fillTimestamps(input);
    }
  }

  // 3. Resolve output directory
  const outputDir = resolveOutputDir(opts.outputDir, ctx.stateDir);
  logger.info(`${TAG} Output directory: ${outputDir}`);

  // 4. Check for existing directory / checkpoint (resume detection)
  if (fs.existsSync(outputDir)) {
    const checkpointPath = path.join(outputDir, ".metadata", "checkpoint.json");
    if (fs.existsSync(checkpointPath)) {
      // Checkpoint exists → resume scenario → P0 not implemented
      console.error(
        "\n❌ Resume from checkpoint is not implemented in P0 yet. " +
        "Please use a new output directory.\n" +
        `   Existing: ${outputDir}\n`,
      );
      process.exit(1);
    }

    // Directory exists but no checkpoint → might have stale data
    const entries = fs.readdirSync(outputDir);
    if (entries.length > 0) {
      console.error(
        `\n❌ Output directory already exists and is not empty: ${outputDir}\n` +
        "   Please use a new directory or clean the existing one.\n",
      );
      process.exit(1);
    }
  }

  // 5. Execute seed pipeline
  console.log(`\n🔧 Output: ${outputDir}`);
  console.log(`▶️  Starting seed pipeline...\n`);

  const summary = await executeSeed(input, {
    outputDir,
    openclawConfig: ctx.config,
    pluginConfig: mergedPluginConfig,
    inputFile: opts.input,
    logger,
    onProgress: (progress) => {
      const pct = ((progress.currentRound / progress.totalRounds) * 100).toFixed(0);
      process.stdout.write(
        `\r  [${progress.currentRound}/${progress.totalRounds}] ${pct}% ` +
        `session=${progress.sessionKey} stage=${progress.stage}    `,
      );
    },
  });

  // 6. Print summary
  console.log("\n");
  console.log("╔══════════════════════════════════════════╗");
  console.log("║               Seed Summary               ║");
  console.log("╠══════════════════════════════════════════╣");
  console.log(`║  Sessions:    ${String(summary.sessionsProcessed).padStart(11)}               ║`);
  console.log(`║  Rounds:      ${String(summary.roundsProcessed).padStart(11)}               ║`);
  console.log(`║  Messages:    ${String(summary.messagesProcessed).padStart(11)}               ║`);
  console.log(`║  L0 recorded: ${String(summary.l0RecordedCount).padStart(11)}               ║`);
  console.log(`║  Duration:    ${(summary.durationMs / 1000).toFixed(1).padStart(10)}s               ║`);
  console.log("╚══════════════════════════════════════════╝");
  console.log(`\n📁 Output: ${summary.outputDir}\n`);
}

// ============================
// Helpers
// ============================

/**
 * Load an optional config override file and deep-merge it on top of the
 * base plugin config from openclaw.json.
 *
 * Returns the merged config, or the base config unchanged if no override
 * file is specified.
 */
function loadAndMergePluginConfig(
  base: Record<string, unknown> | undefined,
  configFile: string | undefined,
  logger: { info: (msg: string) => void },
): Record<string, unknown> | undefined {
  if (!configFile) return base;

  const resolved = path.resolve(configFile);
  if (!fs.existsSync(resolved)) {
    console.error(`\n❌ Config override file not found: ${resolved}\n`);
    process.exit(1);
  }

  let override: Record<string, unknown>;
  try {
    const raw = fs.readFileSync(resolved, "utf-8");
    override = JSON.parse(raw) as Record<string, unknown>;
  } catch (err) {
    console.error(
      `\n❌ Failed to parse config override file: ${resolved}\n` +
      `   ${err instanceof Error ? err.message : String(err)}\n`,
    );
    process.exit(1);
  }

  if (typeof override !== "object" || override === null || Array.isArray(override)) {
    console.error(`\n❌ Config override file must contain a JSON object: ${resolved}\n`);
    process.exit(1);
  }

  logger.info(`${TAG} Config override loaded from: ${resolved}`);
  return deepMerge(base ?? {}, override);
}

/**
 * Simple two-level deep merge: for each key in `override`, if both base
 * and override values are plain objects, merge them; otherwise override wins.
 *
 * This is sufficient for the memory-tdai config shape:
 *   { capture: {...}, extraction: {...}, pipeline: {...}, ... }
 */
function deepMerge(
  base: Record<string, unknown>,
  override: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = { ...base };

  for (const key of Object.keys(override)) {
    const baseVal = base[key];
    const overVal = override[key];

    if (isPlainObject(baseVal) && isPlainObject(overVal)) {
      result[key] = { ...baseVal, ...overVal };
    } else {
      result[key] = overVal;
    }
  }

  return result;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function resolveOutputDir(explicit: string | undefined, stateDir: string): string {
  if (explicit) return path.resolve(explicit);

  // Default: <stateDir>/memory-tdai-seed-<YYYYMMDD-HHmmss>
  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const ts =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

  return path.join(stateDir, `memory-tdai-seed-${ts}`);
}

function askConfirmation(prompt: string): Promise<boolean> {
  return new Promise((resolve) => {
    // Delay slightly to let async plugin logs flush before showing the prompt.
    // Without this, the prompt gets buried under registration logs.
    setTimeout(() => {
      console.log("\n" + "─".repeat(60));
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });
      rl.question(`⚠️  ${prompt}`, (answer) => {
        rl.close();
        resolve(answer.trim().toLowerCase() === "y");
      });
    }, 2000);
  });
}

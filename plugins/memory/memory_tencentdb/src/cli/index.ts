/**
 * memory-tdai CLI entry point.
 *
 * Registers the `memory-tdai` namespace under the OpenClaw CLI and
 * wires up all subcommands (currently: `seed`).
 *
 * Registration path:
 *   index.ts → api.registerCli() → registerMemoryTdaiCli() → registerSeedCommand()
 */

import type { Command } from "commander";
import { registerSeedCommand } from "./commands/seed.js";

// ============================
// Context type
// ============================

/**
 * Minimal context needed by seed CLI commands.
 *
 * Derived from OpenClawPluginCliContext but scoped to what seed actually needs,
 * avoiding a hard dependency on the full plugin CLI context type.
 */
export interface SeedCliContext {
  /** OpenClaw config (for LLM calls in L1 extraction). */
  config: unknown;
  /** Raw plugin config (same shape as api.pluginConfig). */
  pluginConfig: unknown;
  /** State directory root (e.g. ~/.openclaw). */
  stateDir: string;
  /** Logger instance. */
  logger: {
    debug?: (message: string) => void;
    info: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
  };
}

// ============================
// Top-level registration
// ============================

/**
 * Register all memory-tdai CLI subcommands under the given Commander program.
 *
 * This function is called by the plugin's `api.registerCli()` registrar.
 * It creates the `memory-tdai` namespace and delegates to individual
 * command registrars.
 *
 * @param program - The `memory-tdai` Commander command (already created by the registrar)
 * @param ctx - CLI context with config, state dir, and logger
 */
export function registerMemoryTdaiCli(program: Command, ctx: SeedCliContext): void {
  // Register subcommands
  registerSeedCommand(program, ctx);

  // Future: registerQueryCommand(program, ctx);
  // Future: registerStatsCommand(program, ctx);
}

/**
 * OpenClawHostAdapter — translates OpenClaw's plugin API into TDAI Core's
 * unified HostAdapter interface.
 *
 * This is the "thin shell" that keeps OpenClaw-specific dependencies
 * (OpenClawPluginApi, pluginConfig, resolveStateDir, event system)
 * confined to the adapter layer while TDAI Core remains host-neutral.
 *
 * Usage (in index.ts):
 *   const adapter = new OpenClawHostAdapter({ api, pluginDataDir, config });
 *   const core = new TdaiCore({ hostAdapter: adapter, config: parsedConfig });
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk/core";
import { OpenClawLLMRunnerFactory } from "./llm-runner.js";
import type {
  HostAdapter,
  RuntimeContext,
  Logger,
  LLMRunnerFactory,
} from "../../core/types.js";

// ============================
// Options
// ============================

export interface OpenClawHostAdapterOptions {
  /** OpenClaw plugin API instance. */
  api: OpenClawPluginApi;
  /** Resolved plugin data directory (e.g. ~/.openclaw/state/memory-tdai). */
  pluginDataDir: string;
  /** Parsed OpenClaw config (for LLM model resolution). */
  openclawConfig: unknown;
}

// ============================
// OpenClawHostAdapter
// ============================

export class OpenClawHostAdapter implements HostAdapter {
  readonly hostType = "openclaw" as const;

  private api: OpenClawPluginApi;
  private pluginDataDir: string;
  private openclawConfig: unknown;
  private runnerFactory: OpenClawLLMRunnerFactory;

  constructor(opts: OpenClawHostAdapterOptions) {
    this.api = opts.api;
    this.pluginDataDir = opts.pluginDataDir;
    this.openclawConfig = opts.openclawConfig;

    this.runnerFactory = new OpenClawLLMRunnerFactory({
      config: opts.openclawConfig,
      agentRuntime: opts.api.runtime.agent,
      logger: opts.api.logger,
    });
  }

  /**
   * Build a RuntimeContext from the current OpenClaw session.
   *
   * In OpenClaw, sessionKey and sessionId come from the event/ctx objects
   * passed to hooks. This method returns a context with sensible defaults;
   * callers can override sessionKey/sessionId per-hook invocation using
   * `buildRuntimeContextForSession()`.
   */
  getRuntimeContext(): RuntimeContext {
    return {
      userId: "default_user",
      sessionId: "",
      sessionKey: "",
      platform: "openclaw",
      workspaceDir: process.cwd(),
      dataDir: this.pluginDataDir,
    };
  }

  /**
   * Build a RuntimeContext for a specific session (used per-hook).
   *
   * This is an OpenClaw-specific convenience that merges session-level
   * identifiers from hook ctx into the base context.
   */
  buildRuntimeContextForSession(sessionKey: string, sessionId?: string): RuntimeContext {
    return {
      ...this.getRuntimeContext(),
      sessionKey,
      sessionId: sessionId ?? "",
    };
  }

  getLogger(): Logger {
    return this.api.logger;
  }

  getLLMRunnerFactory(): LLMRunnerFactory {
    return this.runnerFactory;
  }

  // -- OpenClaw-specific accessors (for index.ts bridge) --------------------

  /** Get the raw OpenClaw plugin API (for legacy callers during migration). */
  getPluginApi(): OpenClawPluginApi {
    return this.api;
  }

  /** Get the OpenClaw config object (for legacy callers during migration). */
  getOpenClawConfig(): unknown {
    return this.openclawConfig;
  }

  /** Get the resolved plugin data directory. */
  getPluginDataDir(): string {
    return this.pluginDataDir;
  }
}

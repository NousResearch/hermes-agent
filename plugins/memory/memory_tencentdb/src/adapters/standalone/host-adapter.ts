/**
 * StandaloneHostAdapter — HostAdapter for the TDAI Gateway (Hermes sidecar).
 *
 * Does NOT depend on OpenClaw. Context is constructed from Gateway config
 * and per-request parameters (session_id, user_id, etc.).
 */

import { StandaloneLLMRunnerFactory } from "./llm-runner.js";
import type { StandaloneLLMConfig } from "./llm-runner.js";
import type {
  HostAdapter,
  RuntimeContext,
  Logger,
  LLMRunnerFactory,
} from "../../core/types.js";

// ============================
// Options
// ============================

export interface StandaloneHostAdapterOptions {
  /** Base data directory for TDAI storage. */
  dataDir: string;
  /** LLM configuration for model calls. */
  llmConfig: StandaloneLLMConfig;
  /** Logger instance. */
  logger: Logger;
  /** Default user ID (can be overridden per-request). */
  defaultUserId?: string;
  /** Platform identifier. */
  platform?: string;
}

// ============================
// StandaloneHostAdapter
// ============================

export class StandaloneHostAdapter implements HostAdapter {
  readonly hostType = "standalone" as const;

  private dataDir: string;
  private logger: Logger;
  private runnerFactory: StandaloneLLMRunnerFactory;
  private defaultUserId: string;
  private platform: string;

  constructor(opts: StandaloneHostAdapterOptions) {
    this.dataDir = opts.dataDir;
    this.logger = opts.logger;
    this.defaultUserId = opts.defaultUserId ?? "default_user";
    this.platform = opts.platform ?? "gateway";

    this.runnerFactory = new StandaloneLLMRunnerFactory({
      config: opts.llmConfig,
      logger: opts.logger,
    });
  }

  getRuntimeContext(): RuntimeContext {
    return {
      userId: this.defaultUserId,
      sessionId: "",
      sessionKey: "",
      platform: this.platform,
      workspaceDir: this.dataDir,
      dataDir: this.dataDir,
    };
  }

  /**
   * Build a RuntimeContext for a specific request.
   * Used by Gateway route handlers to scope each request to the correct user/session.
   */
  buildRuntimeContextForRequest(params: {
    userId?: string;
    sessionId?: string;
    sessionKey?: string;
    platform?: string;
  }): RuntimeContext {
    return {
      userId: params.userId ?? this.defaultUserId,
      sessionId: params.sessionId ?? "",
      sessionKey: params.sessionKey ?? params.sessionId ?? "",
      platform: params.platform ?? this.platform,
      workspaceDir: this.dataDir,
      dataDir: this.dataDir,
    };
  }

  getLogger(): Logger {
    return this.logger;
  }

  getLLMRunnerFactory(): LLMRunnerFactory {
    return this.runnerFactory;
  }
}

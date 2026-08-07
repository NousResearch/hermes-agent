/**
 * PersonaTrigger: determines whether to trigger persona generation.
 * Implements the 5 trigger conditions from the legacy system.
 */

import { CheckpointManager } from "../../utils/checkpoint.js";
import { stripSceneNavigation } from "../scene/scene-navigation.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";

import type { Logger } from "../types.js";

const TAG = "[memory-tdai] [trigger]";

type TriggerLogger = Logger;

export interface TriggerResult {
  should: boolean;
  reason: string;
}

export class PersonaTrigger {
  private dataDir: string;
  private interval: number;
  private logger: TriggerLogger | undefined;
  private storage: StorageAdapter | undefined;

  constructor(opts: { dataDir: string; interval: number; logger?: TriggerLogger; storage?: StorageAdapter }) {
    this.dataDir = opts.dataDir;
    this.interval = opts.interval;
    this.logger = opts.logger;
    this.storage = opts.storage;
  }

  async shouldGenerate(): Promise<TriggerResult> {
    const cpManager = new CheckpointManager(this.dataDir, undefined, this.storage);
    const cp = await cpManager.read();
    this.logger?.debug?.(`${TAG} Evaluating: total_processed=${cp.total_processed}, last_persona_at=${cp.last_persona_at}, memories_since=${cp.memories_since_last_persona}, scenes=${cp.scenes_processed}`);

    // Priority 1: Agent explicitly requested persona update
    if (cp.request_persona_update) {
      const result: TriggerResult = {
        should: true,
        reason: `主动请求: ${cp.persona_update_reason || "Agent 请求更新"}`,
      };
      this.logger?.debug?.(`${TAG} Trigger P1 (explicit request): ${result.reason}`);
      return result;
    }

    const hasSceneFiles = await this.hasSceneFiles();
    const hasPersonaBody = await this.hasPersonaBody();
    const hasGeneratedPersona = cp.last_persona_at > 0 || cp.last_persona_time !== "" || hasPersonaBody;

    // Priority 2: Cold start — first extraction done, no persona yet, has scene files
    if (
      cp.scenes_processed > 0 &&
      !hasGeneratedPersona &&
      hasSceneFiles
    ) {
      const result: TriggerResult = { should: true, reason: "首次冷启动：首次提取完成且有场景文件" };
      this.logger?.debug?.(`${TAG} Trigger P2 (cold start): scenes_processed=${cp.scenes_processed}, total_processed=${cp.total_processed}`);
      return result;
    }

    // Priority 2.5: Recovery — persona was generated before but persona.md body
    // is now empty (corrupted/missing). Regenerate to restore.
    if (
      hasGeneratedPersona &&
      hasSceneFiles &&
      !hasPersonaBody
    ) {
      const result: TriggerResult = { should: true, reason: "恢复：persona.md 正文丢失或为空，需要重新生成" };
      this.logger?.debug?.(`${TAG} Trigger P2.5 (recovery): last_persona_time=${cp.last_persona_time || "(empty)"}, persona body missing`);
      return result;
    }

    // Priority 3: First scene block extraction
    if (cp.scenes_processed === 1 && cp.memories_since_last_persona > 0) {
      const result: TriggerResult = { should: true, reason: "首次 Scene Block 提取完成" };
      this.logger?.debug?.(`${TAG} Trigger P3 (first scene): scenes_processed=${cp.scenes_processed}`);
      return result;
    }

    // Priority 4: Reached threshold
    if (cp.memories_since_last_persona >= this.interval) {
      const result: TriggerResult = {
        should: true,
        reason: `达到阈值: ${cp.memories_since_last_persona} >= ${this.interval}`,
      };
      this.logger?.debug?.(`${TAG} Trigger P4 (threshold): ${result.reason}`);
      return result;
    }

    this.logger?.debug?.(`${TAG} No trigger conditions met`);
    return { should: false, reason: "" };
  }

  private async hasSceneFiles(): Promise<boolean> {
    try {
      if (this.storage) {
        const files = await this.storage.readdirNames(StoragePaths.sceneBlocksDir, ".md");
        return files.length > 0;
      }
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const blocksDir = path.default.join(this.dataDir, "scene_blocks");
      const files = await fs.default.readdir(blocksDir);
      return files.some((f) => f.endsWith(".md"));
    } catch {
      return false;
    }
  }

  /**
   * Check whether persona.md has a non-empty body (excluding scene navigation).
   * Returns false if the file doesn't exist, is empty, or only contains
   * scene navigation (no actual persona content).
   */
  private async hasPersonaBody(): Promise<boolean> {
    try {
      let raw: string | null;
      if (this.storage) {
        raw = await this.storage.readFile(StoragePaths.persona);
      } else {
        const fs = await import("node:fs/promises");
        const path = await import("node:path");
        raw = await fs.default.readFile(path.default.join(this.dataDir, "persona.md"), "utf-8");
      }
      if (!raw) return false;
      const body = stripSceneNavigation(raw).trim();
      return body.length > 0;
    } catch {
      return false;
    }
  }
}

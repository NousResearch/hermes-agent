/**
 * skill-versioning — 版本递增的事务编排
 *
 * 把 skill-store 的 `appendVersion` 与 storage 的 `copyTree` 包成一组「append a new version」原语。
 * 为 SkillCore (Phase 6) 简化 6 个 manage action 的实现。
 *
 * 一次完整的"加一版"动作：
 *   1. 取 head（如果有）
 *   2. 校验：owner / version / 内容是否变更
 *   3. storage 拷贝 head 的版本目录到新版本目录（如果有 head）
 *   4. 在新版本目录上 apply 本次资源变更（write/remove）
 *   5. store.appendVersion 写 DB（事务内）
 *   6. 失败时尝试清理已写入的 storage 副本（best-effort）
 */

import { createHash } from "node:crypto";

import type { ISkillStore } from "./skill-store.interface.js";
import { IdempotentNoOpError, SkillStoreError } from "./skill-store.js";
import { SkillResourceStore, SkillResourceError, type SkillResourcePayload } from "./skill-resource-store.js";
import type { StorageAdapter } from "../storage/adapter.js";
import type { SkillManifestEntry, Skill } from "./types.js";

function computeContentHash(content: string): string {
  return createHash("md5").update(content, "utf-8").digest("hex");
}

export interface AppendVersionContext {
  user_id?: string;
  team_id?: string;
  agent_id?: string;
  task_id?: string;
}

export interface AppendVersionMutation {
  /** SKILL.md 全文（含 frontmatter）。所有写入路径都需要它（hash 校验幂等）。 */
  content: string;
  /** 解析后 frontmatter 提取出的 name / description。 */
  name: string;
  description: string;
  /** 资源变更：本次新增 / 覆盖的文件。 */
  resourcesToWrite?: SkillResourcePayload[];
  /** 资源变更：本次删除的相对 path。 */
  resourcesToRemove?: string[];
  /** 可选：metadata_json 直接覆盖（默认保留上一版）。 */
  metadata_json?: string;
}

export interface SkillVersioningOptions {
  store: ISkillStore;
  resources: SkillResourceStore;
  storage: StorageAdapter;
  logger?: { info(msg: string): void; warn(msg: string): void; error(msg: string): void };
  /**
   * VDB 条数变更回调（用于向 Shark 上报用量）。
   * 每次 store.appendVersion / store.deleteVersion 成功后触发。
   * @param delta +1 表示新增了一条 VDB 文档，-N 表示删除了 N 条。
   */
  onSkillVdbChanged?: (delta: number) => void;
  /**
   * v1 首创时的资产登记钩子。
   *
   * 契约：
   *  - 在 storage 与 DB 写入之前 `await` 调用（前置一致性保护）
   *  - 抛异常 = createNewSkill 失败，storage / DB 都不会写
   *  - 上层实现须幂等（同一 skill_id 多次调用应成功且不产生副作用）
   *  - 仅 `createNewSkill` 触发；`appendNextVersion` / TTL 清理均不触发
   *    （asset 与 version 无关，只认 skill_id）
   */
  onSkillCreated?: (params: {
    skill_id: string;
    team_id?: string;
    agent_id?: string;
    user_id?: string;
    name: string;
    description: string;
  }) => Promise<void>;
}

export class SkillVersioning {
  private readonly store: ISkillStore;
  private readonly resources: SkillResourceStore;
  private readonly storage: StorageAdapter;
  private readonly logger?: SkillVersioningOptions["logger"];
  private readonly onSkillVdbChanged?: (delta: number) => void;
  private readonly onSkillCreated?: SkillVersioningOptions["onSkillCreated"];

  constructor(opts: SkillVersioningOptions) {
    this.store = opts.store;
    this.resources = opts.resources;
    this.storage = opts.storage;
    this.logger = opts.logger;
    this.onSkillVdbChanged = opts.onSkillVdbChanged;
    this.onSkillCreated = opts.onSkillCreated;
  }

  /**
   * 创建一个全新 skill 的 v1。head 不存在；调用方负责生成 skill_id。
   *
   * 跨系统"事务"编排（COS + skill DB + meta_assets）：
   *
   *   1. writeResource → COS               ← 最不可靠，先做；失败零 DB 副作用
   *   2. store.appendVersion → skill DB    ← 失败：反向清 COS（cleanupVersionDir）
   *   3. onSkillCreated → meta_assets      ← 失败：反向删 skill DB (deleteSkill) + 清 COS
   *
   * 为什么这个顺序：跨 3 个系统没有真事务，只能靠"顺序 + 补偿"。原则是
   * **最容易失败的先做、失败零副作用的先做、可靠的收尾**。COS 是最脆的（网络/
   * 认证/权限），skill DB 是本地事务几乎不会失败，asset 涉及 agent 查询/team
   * 校验/多表写但也是本地 DB。原实现"asset 先写"违背了这个原则：曾出现过
   * COS 认证挂 → skill 没落库 → asset 表却已经有一行的孤儿状态。
   *
   * 极端情况（步骤 2/3 rollback 又失败）：
   *   - 孤儿 skill（skill 落库但 asset 缺）由 `onSkillAccessed` 读时自愈补登记，
   *     用户下次 get/readFile 就会补上。闭环存在。
   *   - 孤儿 COS 文件只占空间，读路径全部过 DB，永远不会被误读到。
   */
  async createNewSkill(
    skillId: string,
    ownerAgentId: string,
    ctx: AppendVersionContext,
    mut: AppendVersionMutation,
  ): Promise<Skill> {
    const newVersion = 1;
    const storageDir = this.resources.versionDir(skillId, newVersion);

    // 整 skill 总大小聚合校验（设计 §3.5.1：≤ 50MB）—— 纯计算，零副作用。
    if (mut.resourcesToWrite && mut.resourcesToWrite.length > 0) {
      this.resources.assertTotalSize([], mut.resourcesToWrite, []);
    }

    // ── Step 1: 写 COS（最脆的一步先做，失败零副作用）─────────────────────
    let manifest: SkillManifestEntry[] = [];
    if (mut.resourcesToWrite && mut.resourcesToWrite.length > 0) {
      try {
        for (const p of mut.resourcesToWrite) {
          const entry = await this.resources.writeResource(skillId, newVersion, p);
          manifest.push(entry);
        }
      } catch (e) {
        // 部分文件可能已写，best-effort 清理整个版本目录
        await this.cleanupVersionDir(storageDir).catch(() => { /* ignore */ });
        throw e;
      }
    }

    // ── Step 2: 写 skill DB（本地事务，几乎不会失败；失败反向清 COS）────
    let row: Skill;
    try {
      row = await this.store.appendVersion({
        user_id: ctx.user_id,
        team_id: ctx.team_id,
        agent_id: ctx.agent_id,
        task_id: ctx.task_id,
        skill_id: skillId,
        name: mut.name,
        description: mut.description,
        content: mut.content,
        content_hash: computeContentHash(mut.content),
        manifest,
        storage_dir: storageDir,
        owner_agent_id: ownerAgentId,
        metadata_json: mut.metadata_json,
      });
    } catch (e) {
      await this.cleanupVersionDir(storageDir).catch(() => { /* ignore */ });
      throw e;
    }

    // ── Step 3: 登记 meta_assets（agent/team 校验 + createAsset + bind）──
    //  失败 → 反向删 skill DB (deleteSkill) → 反向清 COS → 抛回业务错误。
    //  onSkillCreated 未注入（如 tdai-core 未挂钩子）→ 直接跳过。
    if (this.onSkillCreated) {
      try {
        await this.onSkillCreated({
          skill_id: skillId,
          team_id: ctx.team_id,
          agent_id: ctx.agent_id,
          user_id: ctx.user_id,
          name: mut.name,
          description: mut.description,
        });
      } catch (assetErr) {
        // 反向删 skill DB。deleteSkill 本身也可能失败（DB 挂了），
        // 但概率极低；即便失败，onSkillAccessed 读时自愈能收敛孤儿 skill。
        // reportVdbDelta=false：+1 从未上报过（步骤 3 之后才上报），rollback
        // 若上报 -1 会让 shark 记账出现负偏差。
        try {
          await this.deleteSkill(skillId, ctx.team_id, { reportVdbDelta: false });
        } catch (rollbackErr) {
          this.logger?.error(
            `[skill-tx] rollback deleteSkill failed for ${skillId} (team=${ctx.team_id ?? "-"}): ` +
              (rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr)),
          );
        }
        // COS 目录 deleteSkill 内部已清（listVersions → rmdir），这里不重复。
        // 但如果 deleteSkill 因 DB 挂掉没跑到 storage 清理，兜底再清一次。
        await this.cleanupVersionDir(storageDir).catch(() => { /* ignore */ });
        throw assetErr;
      }
    }

    // 所有系统写入均成功 → 上报 VDB delta
    this.onSkillVdbChanged?.(1);
    return row;
  }

  /**
   * 在已有 head 之上追加新版本。head 必须存在（调用方先调 store.getHead 拿到）。
   *
   * 内容未变（content_hash 相同）且无资源变更 → 返回 head（幂等，不写 storage / DB）。
   */
  async appendNextVersion(
    head: Skill,
    ctx: AppendVersionContext,
    mut: AppendVersionMutation,
  ): Promise<Skill> {
    const newVersion = head.version + 1;
    const newStorageDir = this.resources.versionDir(head.skill_id, newVersion);
    const oldStorageDir = head.storage_dir;
    const newContentHash = computeContentHash(mut.content);

    const noContentChange = newContentHash === head.content_hash;
    const noResourceChange =
      (!mut.resourcesToWrite || mut.resourcesToWrite.length === 0) &&
      (!mut.resourcesToRemove || mut.resourcesToRemove.length === 0);

    if (noContentChange && noResourceChange) {
      return head; // 幂等
    }

    // 整 skill 总大小聚合校验（设计 §3.5.1：≤ 50MB）。
    if (
      (mut.resourcesToWrite && mut.resourcesToWrite.length > 0) ||
      (mut.resourcesToRemove && mut.resourcesToRemove.length > 0)
    ) {
      this.resources.assertTotalSize(
        head.manifest,
        mut.resourcesToWrite,
        mut.resourcesToRemove,
      );
    }

    // 1) 拷贝旧版本目录到新版本目录（如果旧版本有内容）
    try {
      await this.storage.copyTree(oldStorageDir, newStorageDir);
    } catch (e) {
      const msg = (e as Error).message ?? "";
      // 如果旧版本根本没目录（旧 skill 创建时无资源），跳过 copy
      if (!/STORAGE_NOT_FOUND/.test(msg)) {
        throw e;
      }
    }

    // 2) 在新版本目录上应用资源变更
    let manifest: SkillManifestEntry[] = [...head.manifest];
    try {
      if (mut.resourcesToRemove) {
        for (const p of mut.resourcesToRemove) {
          await this.resources.removeResource(head.skill_id, newVersion, p);
          manifest = manifest.filter((m) => m.path !== p);
        }
      }
      if (mut.resourcesToWrite) {
        for (const p of mut.resourcesToWrite) {
          const entry = await this.resources.writeResource(head.skill_id, newVersion, p);
          // 合并到 manifest（覆盖同 path）
          manifest = manifest.filter((m) => m.path !== entry.path);
          manifest.push(entry);
        }
      }
    } catch (e) {
      await this.cleanupVersionDir(newStorageDir).catch(() => { /* ignore */ });
      throw e;
    }

    // 3) 写 DB（事务内 + fts 同步）
    try {
      const row = await this.store.appendVersion({
        user_id: ctx.user_id,
        team_id: ctx.team_id,
        agent_id: ctx.agent_id,
        task_id: ctx.task_id,
        skill_id: head.skill_id,
        name: mut.name,
        description: mut.description,
        content: mut.content,
        content_hash: newContentHash,
        manifest,
        storage_dir: newStorageDir,
        owner_agent_id: head.owner_agent_id,
        metadata_json: mut.metadata_json ?? head.metadata_json,
      });
      this.onSkillVdbChanged?.(1);
      return row;
    } catch (e) {
      // DB 失败 → 清理刚 copy 的新目录
      // 注：store.appendVersion 不再做 hash 幂等（由本类早期 short-circuit 处理），
      // 因此不会抛 IdempotentNoOpError；类型仍保留导出供外部 import 一处即可。
      await this.cleanupVersionDir(newStorageDir).catch(() => { /* ignore */ });
      throw e;
    }
  }

  // ─────────────────────────────────────────────────
  //  helpers
  // ─────────────────────────────────────────────────

  private async cleanupVersionDir(dir: string): Promise<void> {
    if (!dir) return;
    try { await this.storage.rmdir(dir); } catch { /* ignore */ }
  }

  // ─────────────────────────────────────────────────
  //  真删除：一次性删掉某 skill 的所有版本（含 storage 与 shark 上报）
  // ─────────────────────────────────────────────────

  /**
   * 物理删除一整个 skill（含所有版本行 + 每个版本的 storage 目录）。
   *
   * 编排语义：
   *   1. 先 listVersions 拿到所有版本的 storage_dir（DB 是权威源，先读后删）
   *   2. store.deleteAllVersions —— 一次性 DELETE 所有版本行 + 清 fts / vec
   *   3. 逐版本 rmdir storage（失败仅 warn，不回滚 DB）
   *   4. 汇总一次 `onSkillVdbChanged(-N)`（N = 实际删除的行数）
   *      —— 与 TTL 路径逐行 `onSkillVdbChanged(-1)` 不同，delete 走整块上报，
   *      减少 shark HTTP 请求；语义上都是 shark MemoryDelta 累加。
   *
   * 返回实际删除的行数。skill 不存在时返回 0，不触发上报（避免虚报）。
   *
   * 权限校验（team_id / owner / expected_version）由调用方 SkillCore.delete
   * 完成。本方法不做业务规则校验，只按 (skill_id, team_id) 做物理清理。
   */
  async deleteSkill(
    skillId: string,
    teamId?: string,
    opts?: {
      /**
       * 是否上报 VDB delta。默认 true（正常删除路径）。
       * createNewSkill 的 rollback 路径应传 false —— 因为对应的 +1 从未上报过
       * （+1 只在整个 create 三步全绿之后才发），rollback 里再上报 -N 会让
       * shark 记账出现负偏差。
       */
      reportVdbDelta?: boolean;
    },
  ): Promise<number> {
    const reportDelta = opts?.reportVdbDelta ?? true;

    // 1. 先拉全量版本元信息（拿 storage_dir）。listVersions 上限 1000,
    //    单 skill 版本数在 TTL 保护 + 业务上限下远小于此值。
    const versions = await this.store.listVersions(skillId, teamId, { limit: 1000, offset: 0 });

    // 2. 物理删 DB 行
    const deleted = await this.store.deleteAllVersions(skillId, teamId);
    if (deleted <= 0) {
      // 什么都没删（skill 不存在 / team 不匹配）→ 不上报，不清 storage
      return 0;
    }

    // 3. 清 storage 目录（失败仅 warn）—— 用 listVersions 拿到的 dir，
    //    而不是拼路径，容忍历史版本 storage_dir 命名不一致的情况。
    for (const v of versions) {
      if (!v.storage_dir) continue;
      try {
        await this.storage.rmdir(v.storage_dir);
      } catch {
        this.logger?.warn(`[skill-delete] storage rmdir failed for ${v.storage_dir}`);
      }
    }

    // 4. 一次性上报 shark（-N）；rollback 路径下不上报（见 opts 注释）
    if (reportDelta) {
      this.onSkillVdbChanged?.(-deleted);
    }

    return deleted;
  }

  // ─────────────────────────────────────────────────
  //  TTL：写后清理过期旧版本
  // ─────────────────────────────────────────────────

  private static readonly KEEP_RECENT = 3;

  /**
   * 清理指定 skill 的过期非 head 版本（先删 DB 行，后删 storage 目录）。
   * fire-and-forget 调用，不抛异常。
   */
  async cleanupExpiredVersionsForSkill(
    skillId: string,
    ttlSeconds: number,
    now?: number,
  ): Promise<void> {
    if (ttlSeconds <= 0) return;

    const nowMs = now ?? Date.now();
    const cutoffMs = nowMs - ttlSeconds * 1000;
    const all = await this.store.listVersions(skillId);
    if (!all.length) return;

    // archived skill 整组保护
    const head = all.find((v) => v.is_head);
    if (!head || head.status === "archived") return;

    // version DESC
    const sorted = [...all].sort((a, b) => b.version - a.version);
    // KEEP_RECENT 保护最近 N 个非 head 版本（即使过期）
    const protectedVersions = new Set(
      sorted.filter((v) => !v.is_head).slice(0, SkillVersioning.KEEP_RECENT).map((v) => v.version),
    );

    for (const v of sorted) {
      if (v.is_head) continue;
      if (protectedVersions.has(v.version)) continue;
      if (v.created_at_ms >= cutoffMs) continue;

      // 先删 DB 行（数据源），再删 storage 目录（附属物）
      const deleted = await this.store.deleteVersion(v.skill_id, v.version);
      if (!deleted) continue;

      // 上报 VDB 删除（负值）
      this.onSkillVdbChanged?.(-1);

      // storage 删除失败仅记日志，不影响 DB 的正确性
      try {
        await this.storage.rmdir(v.storage_dir);
      } catch {
        this.logger?.warn(`[skill-ttl] storage rmdir failed for ${v.storage_dir}`);
      }
    }
  }

}

// 重新导出错误类型，让上层 import 一处即可
export { IdempotentNoOpError, SkillStoreError, SkillResourceError };

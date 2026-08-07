import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { IMemoryStore, ProfileRecord, ProfileSyncRecord } from "../store/types.js";
import { readSceneIndex, syncSceneIndex } from "../scene/scene-index.js";
import { generateSceneNavigation, stripSceneNavigation } from "../scene/scene-navigation.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";
import type { Logger } from "../types.js";

export const DEFAULT_PROFILE_SCOPE = "global";

export type ProfileIsolation = { teamId?: string; userId?: string; agentId?: string; sessionId?: string };

export interface ProfileScopeOptions {
  scope?: string;
  isolation?: ProfileIsolation;
}

export function buildProfileIsolationScope(ctx?: ProfileIsolation): string {
  if (!ctx) return DEFAULT_PROFILE_SCOPE;
  const teamId = ctx.teamId || ctx.userId || "default";
  const agentId = ctx.agentId || "default";
  // L2/L3 are team+agent-level memories. L0/L1 keep user/session/task-level
  // isolation, but profiles intentionally ignore userId/sessionId/taskId so
  // one team's agent memory can accumulate across multiple sessions/users.
  return `team:${teamId}|agent:${agentId}`;
}

export function parseProfileIsolationScope(scope: string): ProfileIsolation | undefined {
  const parts = scope.split("|");
  const values: Record<string, string> = {};
  for (const part of parts) {
    const idx = part.indexOf(":");
    if (idx <= 0) return undefined;
    values[part.slice(0, idx)] = part.slice(idx + 1);
  }
  if (!values.agent) return undefined;
  const sessionId = values.session ? safeDecodeURIComponent(values.session) : undefined;
  if (values.team) return { teamId: values.team, agentId: values.agent, ...(sessionId ? { sessionId } : {}) };
  if (values.user) return { userId: values.user, agentId: values.agent, ...(sessionId ? { sessionId } : {}) };
  return undefined;
}

function safeDecodeURIComponent(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function resolveProfileScope(options?: ProfileScopeOptions): { scope: string; isolation?: ProfileIsolation } {
  const scope = options?.scope ?? buildProfileIsolationScope(options?.isolation);
  return { scope, isolation: options?.isolation ?? parseProfileIsolationScope(scope) };
}

function profileMatchesScope(record: ProfileRecord, scope: string, isolation?: ProfileIsolation): boolean {
  if (scope === DEFAULT_PROFILE_SCOPE && !isolation) return true;
  if (record.id === buildProfileStableId(scope, record.type, record.filename)) return true;
  if (!isolation) return false;
  if (isolation.teamId) {
    return record.teamId === isolation.teamId && record.agentId === isolation.agentId;
  }
  return record.userId === isolation.userId
    && record.agentId === isolation.agentId;
}

/** Check if an error is a rename race condition (another concurrent pull won). */
function isRenameRaceError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException)?.code;
  return code === "ENOTEMPTY" || code === "EEXIST";
}

export interface ProfileBaseline {
  version: number;
  contentMd5: string;
  createdAtMs: number;
}

export function buildProfileStableId(scope: string, type: "l2" | "l3", filename: string): string {
  const hash = createHash("sha256")
    .update(`${scope}\u0000${type}\u0000${filename}`)
    .digest("hex");
  return `profile:v1:${hash}`;
}

function md5(text: string): string {
  return createHash("md5").update(text).digest("hex");
}

async function statTimes(filePath: string): Promise<{ createdAtMs: number; updatedAtMs: number }> {
  try {
    const stat = await fs.stat(filePath);
    return {
      createdAtMs: Math.floor(stat.birthtimeMs || stat.ctimeMs || Date.now()),
      updatedAtMs: Math.floor(stat.mtimeMs || Date.now()),
    };
  } catch {
    const now = Date.now();
    return { createdAtMs: now, updatedAtMs: now };
  }
}

async function refreshPersonaNavigation(dataDir: string, storage?: StorageAdapter): Promise<void> {
  // Read persona body
  let body: string;
  if (storage) {
    const raw = await storage.readFile(StoragePaths.persona);
    if (!raw) return;
    body = stripSceneNavigation(raw).trim();
  } else {
    const personaPath = path.join(dataDir, "persona.md");
    try {
      body = stripSceneNavigation(await fs.readFile(personaPath, "utf-8")).trim();
    } catch {
      return;
    }
  }
  if (!body) return;

  const index = await readSceneIndex(dataDir, storage);
  const nav = generateSceneNavigation(index, undefined, false);
  const finalContent = nav ? `${body}\n\n${nav}\n` : `${body}\n`;

  if (storage) {
    await storage.writeFile(StoragePaths.persona, finalContent);
  } else {
    await fs.writeFile(path.join(dataDir, "persona.md"), finalContent, "utf-8");
  }
}

export async function listLocalProfiles(
  dataDir: string,
  storage?: StorageAdapter,
  options?: ProfileScopeOptions,
): Promise<ProfileRecord[]> {
  const profiles: ProfileRecord[] = [];
  const { scope, isolation } = resolveProfileScope(options);

  // ── List L2 scene blocks ──
  if (storage) {
    try {
      const files = (await storage.readdirNames(StoragePaths.sceneBlocksDir, ".md")).sort();
      for (const filename of files) {
        const content = await storage.readFile(`${StoragePaths.sceneBlocksDir}${filename}`);
        if (content === null) continue;
        const stat = await storage.stat(`${StoragePaths.sceneBlocksDir}${filename}`);
        const now = Date.now();
        const createdAtMs = stat?.createdAt ?? now;
        const updatedAtMs = stat?.lastModified ?? now;
        profiles.push({
          id: buildProfileStableId(scope, "l2", filename),
          type: "l2",
          filename,
          content,
          contentMd5: md5(content),
          teamId: isolation?.teamId,
          userId: isolation?.userId,
          agentId: isolation?.agentId,
          sessionId: undefined,
          version: 0,
          createdAtMs,
          updatedAtMs,
        });
      }
    } catch {
      // ignore missing scene_blocks
    }
  } else {
    const blocksDir = path.join(dataDir, "scene_blocks");
    try {
      const files = (await fs.readdir(blocksDir)).filter((file) => file.endsWith(".md")).sort();
      for (const filename of files) {
        const filePath = path.join(blocksDir, filename);
        const content = await fs.readFile(filePath, "utf-8");
        const { createdAtMs, updatedAtMs } = await statTimes(filePath);
        profiles.push({
          id: buildProfileStableId(scope, "l2", filename),
          type: "l2",
          filename,
          content,
          contentMd5: md5(content),
          teamId: isolation?.teamId,
          userId: isolation?.userId,
          agentId: isolation?.agentId,
          sessionId: undefined,
          version: 0,
          createdAtMs,
          updatedAtMs,
        });
      }
    } catch {
      // ignore missing scene_blocks directory
    }
  }

  // ── List L3 documents ──
  const l3Files = [StoragePaths.persona];
  if (storage) {
    for (const filename of l3Files) {
      try {
        const raw = await storage.readFile(filename);
        if (!raw) continue;
        const body = stripSceneNavigation(raw).trim();
        if (!body) continue;
        const stat = await storage.stat(filename);
        const now = Date.now();
        profiles.push({
          id: buildProfileStableId(scope, "l3", filename),
          type: "l3",
          filename,
          content: body,
          contentMd5: md5(body),
          teamId: isolation?.teamId,
          userId: isolation?.userId,
          agentId: isolation?.agentId,
          sessionId: undefined,
          version: 0,
          createdAtMs: stat?.createdAt ?? now,
          updatedAtMs: stat?.lastModified ?? now,
        });
      } catch {
        // ignore missing L3 document
      }
    }
  } else {
    for (const filename of l3Files) {
      const filePath = path.join(dataDir, filename);
      try {
        const raw = await fs.readFile(filePath, "utf-8");
        const body = stripSceneNavigation(raw).trim();
        if (!body) continue;
        const { createdAtMs, updatedAtMs } = await statTimes(filePath);
        profiles.push({
          id: buildProfileStableId(scope, "l3", filename),
          type: "l3",
          filename,
          content: body,
          contentMd5: md5(body),
          teamId: isolation?.teamId,
          userId: isolation?.userId,
          agentId: isolation?.agentId,
          sessionId: undefined,
          version: 0,
          createdAtMs,
          updatedAtMs,
        });
      } catch {
        // ignore missing L3 document
      }
    }
  }

  return profiles;
}

export async function pullProfilesToLocal(
  dataDir: string,
  store: IMemoryStore,
  logger: Logger,
  storage?: StorageAdapter,
  options?: ProfileScopeOptions,
): Promise<Map<string, ProfileBaseline>> {
  if (!store.pullProfiles) return new Map();

  const { scope, isolation } = resolveProfileScope(options);
  const allRecords = await store.pullProfiles();
  const records = allRecords.filter((record) => profileMatchesScope(record, scope, isolation));
  const baseline = new Map<string, ProfileBaseline>();

  // ── Storage-backed path (COS / abstracted backend) ──
  // No atomic rename available — write each file individually and rely on
  // the backend's eventual consistency. Concurrent pulls write the same
  // remote snapshot so last-writer-wins is acceptable.
  //
  // Deletion semantics: the local copy of a profile is only removed when the
  // remote `records` list does NOT contain a matching entry. An MD5 mismatch
  // (data integrity check failure) means we *skip the write* and *keep the
  // local copy* — the local data is still our best snapshot, and a transient
  // remote corruption must never cascade into local data loss. Treat the
  // record as "present but unreadable this round" and let a future sync
  // self-heal.
  if (storage) {
    // Track which remote records exist (regardless of whether we could write
    // them). Anything not in these sets has been deleted upstream.
    const remoteL2Files = new Set<string>();
    let remoteHasPersona = false;

    for (const record of records) {
      baseline.set(record.id, {
        version: record.version,
        contentMd5: record.contentMd5,
        createdAtMs: record.createdAtMs,
      });

      if (record.type === "l2") {
        // Mark presence first so a corrupted L2 record does not look "deleted"
        // and trigger an unintended unlink below.
        remoteL2Files.add(record.filename);
        if (md5(record.content) !== record.contentMd5) {
          logger.debug?.(`[memory-tdai][profile-sync] MD5 mismatch for ${record.filename} (skip write, keep local)`);
          continue;
        }
        await storage.writeFile(`${StoragePaths.sceneBlocksDir}${record.filename}`, record.content);
      } else if (record.type === "l3") {
        // Mark presence first so a corrupted L3 record cannot accidentally
        // delete the local persona.md (the previous failure mode).
        remoteHasPersona = true;
        // Verify against the raw stored content (store-side invariant):
        // contentMd5 must equal md5(record.content). After the writer-side
        // fix in handleCoreWrite, both stores keep the stripped+trimmed
        // body, so this is the sole expected shape.
        if (md5(record.content) !== record.contentMd5) {
          logger.debug?.(`[memory-tdai][profile-sync] MD5 mismatch for ${record.filename} (skip write, keep local)`);
          continue;
        }
        // Defensive: tolerate legacy records that may still carry a Scene
        // Navigation footer (written before the fix). Stripping is a no-op
        // for clean records and avoids re-persisting a stale footer.
        const personaBody = stripSceneNavigation(record.content).trim();
        if (!personaBody) continue;
        await storage.writeFile(StoragePaths.persona, personaBody);
      }
    }

    // Delete L2 files that no longer exist remotely
    try {
      const localFiles = await storage.readdirNames(StoragePaths.sceneBlocksDir, ".md");
      for (const filename of localFiles) {
        if (!remoteL2Files.has(filename)) {
          await storage.unlink(`${StoragePaths.sceneBlocksDir}${filename}`);
        }
      }
    } catch { /* ignore */ }

    if (!remoteHasPersona) {
      try { await storage.unlink(StoragePaths.persona); } catch { /* ignore */ }
    }

    await syncSceneIndex(dataDir, storage);
    await refreshPersonaNavigation(dataDir, storage);
    logger.debug?.(`[memory-tdai][profile-sync] Pulled ${records.length} profile(s) to storage`);
    return baseline;
  }

  // ── Local filesystem path (original logic, uses atomic rename via temp dir) ──
  //
  // Same data-loss-safety rule as the storage path: an MD5 mismatch must NOT
  // erase the local copy. Because the fs path swaps the entire scene_blocks/
  // directory and persona.md via rename, "missing in temp" ≠ "deleted on
  // remote" — so we explicitly carry forward existing local copies for any
  // L2 record that failed verification, and we decide persona deletion based
  // on whether the remote `records` list contains a matching l3 record (not
  // on whether the temp file exists).
  const remoteHasL3 = records.some((r) => r.type === "l3");
  const localBlocksDir = path.join(dataDir, "scene_blocks");
  const tempDir = await fs.mkdtemp(path.join(dataDir, ".profiles-pull-"));
  const tempBlocksDir = path.join(tempDir, "scene_blocks");
  await fs.mkdir(tempBlocksDir, { recursive: true });

  try {
    for (const record of records) {
      baseline.set(record.id, {
        version: record.version,
        contentMd5: record.contentMd5,
        createdAtMs: record.createdAtMs,
      });

      if (record.type === "l2") {
        const target = path.join(tempBlocksDir, record.filename);
        if (md5(record.content) !== record.contentMd5) {
          logger.debug?.(`[memory-tdai][profile-sync] MD5 mismatch for ${record.filename} (skip write, keep local)`);
          // Carry forward the existing local copy so the upcoming rename
          // does not wipe it. If there is no local copy yet, leave temp
          // empty for this filename — there is nothing to preserve.
          try {
            await fs.copyFile(path.join(localBlocksDir, record.filename), target);
          } catch { /* no existing local file — fine */ }
          continue;
        }
        await fs.writeFile(target, record.content, "utf-8");
        continue;
      }

      if (record.type === "l3") {
        // Verify against raw stored content (store-side invariant). After the
        // writer-side fix, both stores keep the stripped+trimmed body, so
        // contentMd5 === md5(record.content) is the only expected shape.
        if (md5(record.content) !== record.contentMd5) {
          logger.debug?.(`[memory-tdai][profile-sync] MD5 mismatch for ${record.filename} (skip write, keep local)`);
          continue;
        }
        // Defensive: tolerate legacy records that may still carry a Scene
        // Navigation footer (written before the fix).
        const personaBody = stripSceneNavigation(record.content).trim();
        if (!personaBody) continue;
        await fs.writeFile(path.join(tempDir, "persona.md"), personaBody, "utf-8");
      }
    }

    await fs.rm(localBlocksDir, { recursive: true, force: true });
    await fs.mkdir(path.dirname(localBlocksDir), { recursive: true });
    try {
      await fs.rename(tempBlocksDir, localBlocksDir);
    } catch (err) {
      if (isRenameRaceError(err)) {
        // Another concurrent pull already wrote scene_blocks — ours is redundant.
        // Both pulls fetched the same remote snapshot, so the other result is equivalent.
        logger.debug?.(`[memory-tdai][profile-sync] scene_blocks rename lost race (${(err as NodeJS.ErrnoException).code}), using existing`);
        return baseline;
      }
      throw err;
    }

    const tempPersonaPath = path.join(tempDir, "persona.md");
    const localPersonaPath = path.join(dataDir, "persona.md");
    try {
      await fs.access(tempPersonaPath);
      // Verified persona body present in temp → atomically replace local.
      await fs.rm(localPersonaPath, { force: true });
      try {
        await fs.rename(tempPersonaPath, localPersonaPath);
      } catch (err) {
        if (!isRenameRaceError(err)) throw err;
        logger.debug?.(`[memory-tdai][profile-sync] persona.md rename lost race, using existing`);
      }
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") {
        // No verified persona body in temp. This can mean two very different
        // things — keep the local copy unless we can prove the remote really
        // has no persona record. (Previously this branch always deleted the
        // local file, which destroyed user data on any single MD5 mismatch.)
        if (!remoteHasL3) {
          await fs.rm(localPersonaPath, { force: true });
        } else {
          logger.debug?.(`[memory-tdai][profile-sync] persona.md skipped this round, keeping local copy`);
        }
      } else if (!isRenameRaceError(err)) {
        throw err;
      }
    }

    await syncSceneIndex(dataDir);
    await refreshPersonaNavigation(dataDir);
    logger.debug?.(`[memory-tdai][profile-sync] Pulled ${records.length} profile(s) to local cache`);
    return baseline;
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

export async function syncLocalProfilesToStore(
  dataDir: string,
  store: IMemoryStore,
  baselineMap: Map<string, ProfileBaseline>,
  logger: Logger,
  storage?: StorageAdapter,
  options?: ProfileScopeOptions,
): Promise<void> {
  const localProfiles = await listLocalProfiles(dataDir, storage, options);
  const localIds = new Set(localProfiles.map((profile) => profile.id));

  const syncRecords: ProfileSyncRecord[] = localProfiles
    .filter((profile) => baselineMap.get(profile.id)?.contentMd5 !== profile.contentMd5 || !baselineMap.has(profile.id))
    .map((profile) => ({
      ...profile,
      baselineVersion: baselineMap.get(profile.id)?.version,
    }));

  if (syncRecords.length > 0 && store.syncProfiles) {
    await store.syncProfiles(syncRecords);
    logger.info(`[memory-tdai][profile-sync] Synced ${syncRecords.length} changed profile(s)`);
  }

  const deletedIds = [...baselineMap.keys()].filter((id) => !localIds.has(id));
  if (deletedIds.length > 0 && store.deleteProfiles) {
    await store.deleteProfiles(deletedIds);
    logger.info(`[memory-tdai][profile-sync] Deleted ${deletedIds.length} stale profile(s)`);
  }
}

export async function ensureL2L3Local(
  dataDir: string,
  store: IMemoryStore,
  logger: Logger,
  storage?: StorageAdapter,
  options?: ProfileScopeOptions,
): Promise<Map<string, ProfileBaseline>> {
  if (!store.pullProfiles) return new Map();
  return pullProfilesToLocal(dataDir, store, logger, storage, options);
}

/**
 * Scene Index: maintains a JSON index of all scene blocks for quick lookup.
 */

import { parseSceneBlock } from "./scene-format.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";

export interface SceneIndexEntry {
  filename: string;
  summary: string;
  heat: number;
  created: string;
  updated: string;
}

// ── fs fallback helpers (used when no StorageAdapter is provided) ──

async function fsReadFile(absPath: string): Promise<string | null> {
  const fs = await import("node:fs/promises");
  try {
    return await fs.default.readFile(absPath, "utf-8");
  } catch { return null; }
}

async function fsWriteFile(absPath: string, content: string): Promise<void> {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  await fs.default.mkdir(path.default.dirname(absPath), { recursive: true });
  await fs.default.writeFile(absPath, content, "utf-8");
}

async function fsReaddir(absDir: string, suffix: string): Promise<string[]> {
  const fs = await import("node:fs/promises");
  try {
    const entries = await fs.default.readdir(absDir);
    return entries.filter((f) => f.endsWith(suffix));
  } catch { return []; }
}

/**
 * Read the scene index from disk.
 *
 * The index is written exclusively by syncSceneIndex() (engineering side).
 * The LLM is sandboxed to scene_blocks/ and cannot access this file.
 */
export async function readSceneIndex(dataDir: string, storage?: StorageAdapter): Promise<SceneIndexEntry[]> {
  try {
    let raw: string | null;
    if (storage) {
      raw = await storage.readFile(StoragePaths.sceneIndex);
    } else {
      const path = await import("node:path");
      raw = await fsReadFile(path.default.join(dataDir, ".metadata", "scene_index.json"));
    }
    if (!raw) return [];

    const parsed = JSON.parse(raw) as Array<Record<string, unknown>>;
    if (!Array.isArray(parsed)) return [];

    const entries: SceneIndexEntry[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;

      const filename = typeof item.filename === "string" ? item.filename : "";
      if (!filename) continue;

      entries.push({
        filename,
        summary: typeof item.summary === "string" ? item.summary : "",
        heat: typeof item.heat === "number" ? item.heat : 0,
        created: typeof item.created === "string" ? item.created : "",
        updated: typeof item.updated === "string" ? item.updated : "",
      });
    }
    return entries;
  } catch {
    return [];
  }
}

/**
 * Write the scene index to disk.
 */
export async function writeSceneIndex(
  dataDir: string,
  entries: SceneIndexEntry[],
  storage?: StorageAdapter,
): Promise<void> {
  const content = JSON.stringify(entries, null, 2);
  if (storage) {
    await storage.writeFile(StoragePaths.sceneIndex, content);
  } else {
    const path = await import("node:path");
    await fsWriteFile(path.default.join(dataDir, ".metadata", "scene_index.json"), content);
  }
}

/**
 * Rebuild scene index by scanning all .md files in the scene_blocks directory.
 */
export async function syncSceneIndex(dataDir: string, storage?: StorageAdapter): Promise<SceneIndexEntry[]> {
  let files: string[];
  if (storage) {
    files = await storage.readdirNames(StoragePaths.sceneBlocksDir, ".md");
  } else {
    const path = await import("node:path");
    files = await fsReaddir(path.default.join(dataDir, "scene_blocks"), ".md");
  }

  const entries: SceneIndexEntry[] = [];
  for (const file of files) {
    try {
      let raw: string | null;
      if (storage) {
        raw = await storage.readFile(`${StoragePaths.sceneBlocksDir}${file}`);
      } else {
        const path = await import("node:path");
        raw = await fsReadFile(path.default.join(dataDir, "scene_blocks", file));
      }
      if (!raw) continue;
      const block = parseSceneBlock(raw, file);
      entries.push({
        filename: file,
        summary: block.meta.summary,
        heat: block.meta.heat,
        created: block.meta.created,
        updated: block.meta.updated,
      });
    } catch {
      continue;
    }
  }

  await writeSceneIndex(dataDir, entries, storage);
  return entries;
}

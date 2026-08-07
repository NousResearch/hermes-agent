/**
 * Scene filename normalizer.
 *
 * Defensive engineering layer that runs *after* the LLM writes scene_blocks/*.md
 * and *before* syncSceneIndex(). Even though the prompt forbids spaces and
 * punctuation in filenames, LLMs occasionally produce names like
 * `Daily Rhythm in Shanghai.md`. Such names break:
 *   - Markdown navigation refs that downstream tools parse with `\S+\.md`
 *     (e.g. health-checker's scene reference detection).
 *   - Shell-based tools that iterate scene files without quoting.
 *   - URL/path encoding consumers (COS object keys etc).
 *
 * This module renames offenders to a canonical form on disk and lets every
 * other consumer (PersonaGenerator, recall, profile-sync) read the already
 * sanitized name from scene_index.json — no additional changes needed.
 */

import fs from "node:fs/promises";
import path from "node:path";

/**
 * Normalize a single scene filename.
 *
 * Rules:
 *   - Preserves the `.md` extension (case-insensitive match, lowercased).
 *   - Whitespace runs (spaces / tabs) → single hyphen.
 *   - Strips quotes, brackets, and ASCII punctuation that breaks shell/markdown.
 *   - Collapses consecutive separators (`-`, `_`, `.`).
 *   - Trims leading / trailing separators.
 *   - Falls back to `"scene"` if the stem becomes empty.
 *
 * Allowed character set after normalization (informally):
 *   ASCII alphanumerics, CJK ideographs, hyphen, underscore, dot.
 *
 * Examples:
 *   "Daily Rhythm in Shanghai.md"  → "Daily-Rhythm-in-Shanghai.md"
 *   "日常生活 健康管理.md"          → "日常生活-健康管理.md"
 *   "Coffee (Yirgacheffe).md"      → "Coffee-Yirgacheffe.md"
 *   "  spaced  .md"                → "spaced.md"
 *   ".MD"                          → "scene.md"
 *   "已经规范.md"                   → "已经规范.md" (no-op)
 */
export function normalizeSceneFilename(name: string): string {
  if (!name) return "scene.md";

  // Strip directory components defensively — we only normalize the basename.
  const base = name.replace(/^.*[\\/]/, "");

  // Detect & strip `.md` (case-insensitive). Always re-emit lowercase `.md`.
  const lower = base.toLowerCase();
  const hasMd = lower.endsWith(".md");
  const stem = hasMd ? base.slice(0, -3) : base;

  const safe = stem
    // Replace whitespace runs (incl. NBSP, full-width space) with `-`
    .replace(/[\s\u00A0\u3000]+/g, "-")
    // Drop quotes, brackets, and punctuation known to break shells/markdown.
    // Keep alphanumerics, CJK ideographs, `-`, `_`, `.`.
    .replace(/[()[\]{}<>'"`,;:!?*|/\\=&%$#@^~+]/g, "")
    // Collapse consecutive separators.
    .replace(/-{2,}/g, "-")
    .replace(/_{2,}/g, "_")
    .replace(/\.{2,}/g, ".")
    // Trim leading / trailing separators.
    .replace(/^[-_.]+|[-_.]+$/g, "");

  return (safe || "scene") + ".md";
}

/**
 * Return whether a filename already matches its normalized form.
 * Faster than computing the normalized form when callers only need a yes/no.
 */
export function isNormalizedSceneFilename(name: string): boolean {
  return normalizeSceneFilename(name) === name;
}

/**
 * Resolve a non-conflicting target path inside `dir` for the desired filename.
 *
 * If `desired` (e.g. `Daily-Rhythm.md`) already exists in `dir`, append a
 * numeric suffix `-2`, `-3`, ... before the `.md` extension until a free slot
 * is found. Caller may also pass `excludePath` to ignore a known existing file
 * (e.g. the source path of an in-flight rename, when source != target).
 */
export async function resolveUniqueScenePath(
  dir: string,
  desired: string,
  excludePath?: string,
): Promise<string> {
  const target = path.join(dir, desired);
  if (!(await pathExists(target)) || target === excludePath) return target;

  const ext = ".md";
  const stem = desired.endsWith(ext) ? desired.slice(0, -ext.length) : desired;

  // Bound the search to keep this defensive (LLMs rarely produce hundreds of
  // colliding names; if they do, surface the failure rather than spin).
  for (let i = 2; i < 1000; i++) {
    const candidate = path.join(dir, `${stem}-${i}${ext}`);
    if (!(await pathExists(candidate)) || candidate === excludePath) {
      return candidate;
    }
  }
  throw new Error(
    `resolveUniqueScenePath: could not find a free slot for ${desired} in ${dir} after 1000 attempts`,
  );
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

export interface NormalizeRenameResult {
  /** Number of files that were actually renamed. */
  renamed: number;
  /** Number of files that were already normalized (no-op). */
  skipped: number;
  /** Per-rename audit entries (oldName → newName). */
  renames: Array<{ from: string; to: string }>;
}

/**
 * Walk a scene_blocks directory and rename any `.md` file whose basename does
 * not match `normalizeSceneFilename(basename)`.
 *
 * Safe to call multiple times: subsequent invocations are no-ops once names
 * have stabilized.
 *
 * Notes:
 *   - Non-`.md` files are ignored (the LLM tool surface is restricted to .md,
 *     but the directory may contain transient artifacts).
 *   - Empty / soft-deleted files are not pre-filtered here; the SceneExtractor
 *     cleanup pass handles those before / after this call as appropriate.
 *   - Failures on individual entries are logged via the optional logger and
 *     do not abort the loop — index sync should still see the remaining files.
 */
export async function normalizeSceneFilenames(
  blocksDir: string,
  logger?: { debug?: (m: string) => void; warn?: (m: string) => void },
): Promise<NormalizeRenameResult> {
  const result: NormalizeRenameResult = { renamed: 0, skipped: 0, renames: [] };

  let entries: string[];
  try {
    entries = (await fs.readdir(blocksDir)).filter((f) => f.endsWith(".md"));
  } catch {
    return result;
  }

  for (const file of entries) {
    const normalized = normalizeSceneFilename(file);
    if (normalized === file) {
      result.skipped++;
      continue;
    }

    const from = path.join(blocksDir, file);
    let to: string;
    try {
      to = await resolveUniqueScenePath(blocksDir, normalized, from);
    } catch (err) {
      logger?.warn?.(
        `[filename-normalizer] could not resolve unique target for ${file}: ${err instanceof Error ? err.message : String(err)}`,
      );
      result.skipped++;
      continue;
    }

    if (to === from) {
      // Filesystem already matched (e.g. case-insensitive FS where source and
      // target collapse to the same inode); treat as a no-op.
      result.skipped++;
      continue;
    }

    try {
      await fs.rename(from, to);
      result.renamed++;
      result.renames.push({ from: file, to: path.basename(to) });
      logger?.debug?.(`[filename-normalizer] renamed: ${file} → ${path.basename(to)}`);
    } catch (err) {
      logger?.warn?.(
        `[filename-normalizer] rename failed (${file} → ${path.basename(to)}): ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  return result;
}

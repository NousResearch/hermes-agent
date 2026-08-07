/**
 * read_cos tool: Agent-callable tool for reading files from COS (or local storage).
 *
 * This tool allows the Agent to read Markdown scenario files, persona files,
 * or any other text content stored via IStorageBackend.
 *
 * The Agent provides a full relative key (e.g. "scene_blocks/work/2026Q1.md"
 * or "persona.md"), and the tool returns the file content as text.
 *
 * Path convention (方案 B — 通用文件接口):
 *   - v2 API /scenario/* and /persona/* are semantic interfaces that auto-add
 *     StoragePaths prefixes (e.g. "scene_blocks/"). Users pass short paths.
 *   - This tool is a generic file interface. Users pass the FULL relative key
 *     including the directory prefix. This allows reading any layer's files.
 *
 * Use cases:
 * - Agent reads L2 scenario files: "scene_blocks/work/2026Q1.md"
 * - Agent reads L3 persona file: "persona.md"
 * - Future: Agent reads any stored document
 *
 * The tool is registered via `api.registerTool()` in index.ts.
 */

import type { IStorageBackend, StorageLogger } from "../storage/types.js";

const TAG = "[memory-tencentdb][read_cos]";

// ============================
// Types
// ============================

export interface ReadCosParams {
  /** File path to read, e.g. "scenes/work/2026Q1.md" or "persona/persona.md". */
  path: string;
  /** Optional: encoding hint. Default is "utf-8". */
  encoding?: string;
}

export interface ReadCosResult {
  /** Whether the file was found and read successfully. */
  success: boolean;
  /** File path that was requested. */
  path: string;
  /** File content (text). Empty string if not found. */
  content: string;
  /** File size in bytes. */
  size: number;
  /** Error message if the read failed. */
  error?: string;
}

// ============================
// Tool Implementation
// ============================

/**
 * Execute the read_cos tool: read a file from storage by path.
 *
 * @param params  Tool parameters from the LLM
 * @param storage IStorageBackend instance (injected from plugin context)
 * @param logger  Logger instance
 * @returns ReadCosResult
 */
export async function executeReadCos(
  params: ReadCosParams,
  storage: IStorageBackend,
  logger?: StorageLogger,
): Promise<ReadCosResult> {
  const { path } = params;

  if (!path || typeof path !== "string") {
    return {
      success: false,
      path: path ?? "",
      content: "",
      size: 0,
      error: "Parameter 'path' is required and must be a non-empty string.",
    };
  }

  // Security: prevent path traversal
  if (path.includes("..") || path.startsWith("/")) {
    logger?.warn(`${TAG} Rejected suspicious path: ${path}`);
    return {
      success: false,
      path,
      content: "",
      size: 0,
      error: "Invalid path: must be a relative path without '..'.",
    };
  }

  try {
    logger?.info(`${TAG} [COS_TOOL_CALL] >>> path="${path}" timestamp=${new Date().toISOString()}`);
    const obj = await storage.getObject(path);

    if (!obj) {
      logger?.info(`${TAG} [COS_TOOL_CALL] <<< NOT_FOUND path="${path}"`);
      return {
        success: false,
        path,
        content: "",
        size: 0,
        error: `File not found: ${path}`,
      };
    }

    const encoding = params.encoding ?? "utf-8";
    const content = obj.content.toString(encoding as BufferEncoding);

    logger?.info(`${TAG} [COS_TOOL_CALL] <<< OK path="${path}" size=${obj.size ?? content.length}B`);

    return {
      success: true,
      path,
      content,
      size: obj.size ?? content.length,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logger?.error(`${TAG} Failed to read ${path}: ${message}`);

    return {
      success: false,
      path,
      content: "",
      size: 0,
      error: `Read failed: ${message}`,
    };
  }
}

// ============================
// Tool Schema (for OpenClaw registerTool)
// ============================

/** JSON Schema for the read_cos tool parameters. */
export const READ_COS_TOOL_SCHEMA = {
  type: "object" as const,
  properties: {
    path: {
      type: "string" as const,
      description:
        "Full relative key of the file to read. " +
        "Examples: 'scene_blocks/work/2026Q1.md', 'persona.md'. " +
        "Must be a relative path (no leading slash, no '..').",
    },
  },
  required: ["path"] as const,
};

/** Tool name constant. */
export const READ_COS_TOOL_NAME = "tdai_read_cos";

/** Tool description visible to the LLM. */
export const READ_COS_TOOL_DESCRIPTION =
  "Read a file from the memory storage system by its full relative key. " +
  "Use this to read scenario documents (L2), persona profiles (L3), " +
  "or other stored text files. Returns the file content as text. " +
  "Path examples: 'scene_blocks/my-topic.md', 'persona.md'.";

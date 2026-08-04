/**
 * JSON / Mermaid extraction utilities for LLM response parsing.
 */

/**
 * Extract JSON from raw LLM output. Tolerates markdown fences, extra text.
 */
export function extractJson<T>(raw: string): T | null {
  if (!raw || typeof raw !== "string") return null;

  const trimmed = raw.trim();

  // 1. Direct parse
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    // continue
  }

  // 2. Extract from ```json ... ``` fence
  const jsonFenceMatch = trimmed.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (jsonFenceMatch) {
    try {
      return JSON.parse(jsonFenceMatch[1].trim()) as T;
    } catch {
      // continue
    }
  }

  // 3. Extract first { ... } (greedy last })
  const objStart = trimmed.indexOf("{");
  const objEnd = trimmed.lastIndexOf("}");
  if (objStart !== -1 && objEnd > objStart) {
    try {
      return JSON.parse(trimmed.slice(objStart, objEnd + 1)) as T;
    } catch {
      // continue
    }
  }

  // 4. Extract first [ ... ] (greedy last ])
  const arrStart = trimmed.indexOf("[");
  const arrEnd = trimmed.lastIndexOf("]");
  if (arrStart !== -1 && arrEnd > arrStart) {
    try {
      return JSON.parse(trimmed.slice(arrStart, arrEnd + 1)) as T;
    } catch {
      // continue
    }
  }

  return null;
}

/**
 * Extract mermaid content from a ```mermaid ... ``` code fence.
 */
export function extractMermaidFromFence(raw: string): string | null {
  if (!raw) return null;
  const match = raw.match(/```mermaid\s*\n?([\s\S]*?)```/);
  return match ? match[1].trim() : null;
}

/**
 * Parse JSONL (newline-delimited JSON) string into array.
 * Corrupted lines are silently skipped (logged via optional callback).
 */
export function parseJsonl<T>(
  content: string,
  onBadLine?: (line: string, error: unknown) => void,
): T[] {
  if (!content || !content.trim()) return [];
  const results: T[] = [];
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    try {
      results.push(JSON.parse(trimmed) as T);
    } catch (err) {
      onBadLine?.(trimmed, err);
    }
  }
  return results;
}

/**
 * Serialize array to JSONL string (trailing newline).
 */
export function serializeJsonl<T>(items: T[]): string {
  if (items.length === 0) return "";
  return items.map((item) => JSON.stringify(item)).join("\n") + "\n";
}

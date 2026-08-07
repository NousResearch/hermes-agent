/**
 * skill-format — SKILL.md ↔ SkillFile (de)serialization + validation.
 *
 * Why a dedicated module:
 *   - Keep frontmatter parsing rules in one place (the LLM Reviewer in M6
 *     emits SKILL.md text; the Gateway accepts SKILL.md text in
 *     /v3/skill/create; both flow through here).
 *   - Validation lives next to parsing so callers do `parse → validate`
 *     atomically.
 *
 * Frontmatter contract (SKILL_ENGINEERING_DESIGN §4.1 / §11.1):
 *   - YAML between leading `---\n` ... `\n---` fences (closing `---`
 *     followed by either newline or EOF).
 *   - REQUIRED: name, description.
 *   - OPTIONAL: category, created_at, updated_at, source, resources[].
 *
 * Limits (locked in M3):
 *   - name      : 1..64 chars, ^[a-z0-9][a-z0-9-]*$
 *   - description: 1..1024 chars
 *   - body      : ≤50_000 chars
 *   - resources[*].type ∈ {text, executable, binary}
 */

import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import type { SkillFile } from "./types.js";

const NAME_MAX = 64;
const DESCRIPTION_MAX = 1024;
const BODY_MAX = 50_000;
const NAME_REGEX = /^[a-z0-9][a-z0-9-]*$/;
const RESOURCE_TYPES = new Set(["text", "executable", "binary"]);

/**
 * Parse a SKILL.md raw string into a SkillFile.
 *
 * Accepts CRLF or LF line endings. Throws with a clear message when:
 *   - the file does not start with a `---\n` (no frontmatter)
 *   - the closing `---` fence is missing
 *   - the YAML body cannot be parsed
 *   - the parsed YAML is missing `name` or `description`
 *
 * No length / regex checks here — call `validateSkillFile` next.
 */
export function parseSkillFile(raw: string): SkillFile {
  // Normalize line endings so the fence matcher is simpler.
  const text = raw.replace(/\r\n?/g, "\n");

  if (!text.startsWith("---\n") && text !== "---\n") {
    throw new Error(`[skill][format] missing frontmatter — file must start with '---\\n'`);
  }

  // Find the closing fence. Must be on its own line, after the opening fence.
  // We search starting from index 4 (past the opening "---\n") for "\n---" followed
  // by either "\n" or EOF.
  const start = 4;
  let close = -1;
  let endOfClose = -1;
  for (let i = start; i < text.length - 3; i++) {
    if (text[i] === "\n" && text[i + 1] === "-" && text[i + 2] === "-" && text[i + 3] === "-") {
      // Must be followed by newline or EOF.
      const after = text[i + 4];
      if (after === undefined || after === "\n") {
        close = i + 1; // index of the closing "---"
        endOfClose = i + 4 + (after === "\n" ? 1 : 0);
        break;
      }
    }
  }
  if (close === -1) {
    throw new Error(`[skill][format] missing closing '---' fence in frontmatter`);
  }

  const yamlText = text.slice(start, close);
  let fm: Record<string, unknown>;
  try {
    fm = (parseYaml(yamlText) as Record<string, unknown>) ?? {};
  } catch (err) {
    throw new Error(
      `[skill][format] YAML parse error: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  // Coerce non-string values to string so YAML's type inference (null / true /
  // 123 / comment-like #…) doesn't cause confusing "missing required field"
  // errors when the key is clearly present.  An empty value after coercion is
  // still an error — every SKILL.md MUST carry a non-empty description.
  //
  // Distinguish between "key absent" (undefined) and "key present but non-string"
  // (null / true / 123).  The former is still an error; the latter is coerced.
  const nameRaw = fm.name;
  const name: string = typeof nameRaw === "string"
    ? nameRaw
    : nameRaw === undefined || nameRaw === null
      ? ""
      : String(nameRaw);
  if (name.trim() === "") {
    throw new Error(`[skill][format] frontmatter missing required field 'name'`);
  }
  fm.name = name;

  const descRaw = fm.description;
  const description: string = typeof descRaw === "string"
    ? descRaw
    : descRaw === undefined || descRaw === null
      ? ""
      : String(descRaw);
  if (description.trim() === "") {
    throw new Error(`[skill][format] frontmatter missing required field 'description'`);
  }
  fm.description = description;

  // body is everything after the closing fence; strip a single leading newline
  // so SKILL.md authors can leave a blank line between fence and body.
  let body = text.slice(endOfClose);
  if (body.startsWith("\n")) body = body.slice(1);

  return {
    frontmatter: {
      name: fm.name,
      description: fm.description,
      category: typeof fm.category === "string" ? fm.category : undefined,
      created_at: typeof fm.created_at === "string" ? fm.created_at : undefined,
      updated_at: typeof fm.updated_at === "string" ? fm.updated_at : undefined,
      source: fm.source === "auto" || fm.source === "manual" ? fm.source : undefined,
      resources: parseResources(fm.resources),
    },
    body,
    raw,
  };
}

/**
 * Throw if the SkillFile violates any contract limit.
 * Mutates nothing; pure assertion.
 */
export function validateSkillFile(file: SkillFile): void {
  const { name, description, resources } = file.frontmatter;

  if (!NAME_REGEX.test(name)) {
    throw new Error(
      `[skill][format] invalid name '${name}' — must match ^[a-z0-9][a-z0-9-]*$`,
    );
  }
  if (name.length > NAME_MAX) {
    throw new Error(`[skill][format] name length ${name.length} exceeds max ${NAME_MAX}`);
  }
  if (description.length > DESCRIPTION_MAX) {
    throw new Error(
      `[skill][format] description length ${description.length} exceeds max ${DESCRIPTION_MAX}`,
    );
  }
  if (file.body.length > BODY_MAX) {
    throw new Error(`[skill][format] body length ${file.body.length} exceeds max ${BODY_MAX}`);
  }
  if (resources) {
    for (const r of resources) {
      if (!RESOURCE_TYPES.has(r.type)) {
        throw new Error(
          `[skill][format] resource type '${r.type}' invalid — must be one of text/executable/binary`,
        );
      }
    }
  }
}

/**
 * Serialize a SkillFile back to canonical SKILL.md text (frontmatter + body).
 * Round-trip safe: parseSkillFile(formatSkillFile(f)) yields a file with the
 * same logical fields and body.
 */
export function formatSkillFile(file: SkillFile): string {
  const fm: Record<string, unknown> = {
    name: file.frontmatter.name,
    description: file.frontmatter.description,
  };
  if (file.frontmatter.category !== undefined) fm.category = file.frontmatter.category;
  if (file.frontmatter.created_at !== undefined) fm.created_at = file.frontmatter.created_at;
  if (file.frontmatter.updated_at !== undefined) fm.updated_at = file.frontmatter.updated_at;
  if (file.frontmatter.source !== undefined) fm.source = file.frontmatter.source;
  if (file.frontmatter.resources && file.frontmatter.resources.length > 0) {
    fm.resources = file.frontmatter.resources;
  }
  const yamlBlock = stringifyYaml(fm).replace(/\n+$/, "");
  return `---\n${yamlBlock}\n---\n\n${file.body}`;
}

// ============================
// Internal helpers
// ============================

function parseResources(
  raw: unknown,
): Array<{ path: string; type: "text" | "executable" | "binary" }> | undefined {
  if (!Array.isArray(raw)) return undefined;
  const out: Array<{ path: string; type: "text" | "executable" | "binary" }> = [];
  for (const item of raw) {
    if (typeof item !== "object" || item == null) continue;
    const o = item as Record<string, unknown>;
    if (typeof o.path !== "string" || typeof o.type !== "string") continue;
    // Note: type is preserved verbatim; full enum check happens in validateSkillFile
    // so callers see a clean error message rather than a silent drop.
    out.push({
      path: o.path,
      type: o.type as "text" | "executable" | "binary",
    });
  }
  return out;
}

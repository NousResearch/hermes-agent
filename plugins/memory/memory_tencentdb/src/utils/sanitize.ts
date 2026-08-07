/**
 * Text sanitization for memory pipeline (capture & recall).
 * Removes injected tags, gateway metadata, media noise, etc.
 */

/**
 * Clean text for the memory pipeline: remove injected tags, metadata,
 * timestamps, media markers and base64 image data.
 *
 * Used by both capture (L0 recording) and recall (query cleaning) paths.
 */
export function sanitizeText(text: string): string {
  let cleaned = text;

  // Remove injected memory context tags (prevent feedback loops)
  cleaned = cleaned.replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/g, "");
  cleaned = cleaned.replace(/<user-persona>[\s\S]*?<\/user-persona>/g, "");
  cleaned = cleaned.replace(/<relevant-scenes>[\s\S]*?<\/relevant-scenes>/g, "");
  cleaned = cleaned.replace(/<scene-navigation>[\s\S]*?<\/scene-navigation>/g, "");

  // Remove offload-injected task context blocks (MMD mermaid diagrams)
  cleaned = cleaned.replace(/<current_task_context>[\s\S]*?<\/current_task_context>/g, "");
  cleaned = cleaned.replace(/<history_task_context[\s\S]*?<\/history_task_context>/g, "");

  // Remove framework-injected inbound metadata blocks (from inbound-meta.ts buildInboundUserContextPrefix).
  // These are "label:\n```json\n...\n```" blocks that the framework prepends to user messages.
  // Pattern matches all known block labels:
  //   - Conversation info (untrusted metadata):
  //   - Sender (untrusted metadata):
  //   - Thread starter (untrusted, for context):
  //   - Replied message (untrusted, for context):
  //   - Forwarded message context (untrusted metadata):
  //   - Chat history since last reply (untrusted, for context):
  cleaned = cleaned.replace(
    /(?:Conversation info|Sender|Thread starter|Replied message|Forwarded message context|Chat history since last reply)\s*\(untrusted[\s\S]*?\):\s*```json\s*[\s\S]*?```/g,
    "",
  );

  // Remove conversation metadata JSON blocks (legacy pattern)
  cleaned = cleaned.replace(/```json\s*\{[\s\S]*?"session[\s\S]*?\}\s*```/g, "");

  // Remove framework reply directive tags: [[reply_to_current]], [[reply_to_xxx]], etc.
  cleaned = cleaned.replace(/\[\[reply_to[^\]]*\]\]\s*/g, "");

  // Remove injected skill-selection wrappers, e.g. ¥¥[... ]¥¥
  cleaned = cleaned.replace(/¥¥\[[\s\S]*?\]¥¥/g, "");

  // Remove line-leading timestamps, e.g. "[Tue 2026-03-24 03:48 UTC]"
  // or "[Tue 2026-03-24 20:21 GMT+8]", "[Thu 2026-03-24 01:51 GMT+5:30]"
  // Matches brackets containing word chars, digits, hyphens, colons, plus signs,
  // and spaces — the '+' is needed for timezone offsets like GMT+8, GMT+5:30.
  cleaned = cleaned.replace(/^\[[\w\d\-:+ ]+\]\s*/gm, "");

  // Remove gateway media-attachment markers:
  //   [media attached: /path/to/file.png (image/png) | /path/to/file.png]
  cleaned = cleaned.replace(/\[media attached:[^\]]*\]\s*/g, "");

  // Remove gateway image-reply instructions injected after media attachments.
  // Starts with "To send an image back" and ends before the next real content.
  cleaned = cleaned.replace(
    /To send an image back,[\s\S]*?(?:Keep caption in the text body\.)\s*/g,
    "",
  );

  // Remove "System: [timestamp] Exec completed ..." blocks appended by the framework.
  cleaned = cleaned.replace(/^System:\s*\[[\s\S]*?$/gm, "");

  // Remove inline base64 image data URIs (e.g. data:image/png;base64,iVBOR...)
  // Replace with empty string (not a placeholder) so that pure-image messages
  // become empty after sanitization and are naturally filtered by length checks.
  cleaned = cleaned.replace(/data:image\/[a-z+]+;base64,[A-Za-z0-9+/=]+/gi, "");

  // Remove null chars + compress whitespace
  cleaned = cleaned.replace(/\0/g, "").replace(/\n{3,}/g, "\n\n").trim();

  return cleaned;
}

/**
 * Strip fenced code blocks from assistant replies before L0 capture.
 *
 * AI responses often contain large code snippets (```...```) that dilute
 * the semantic signal for embedding and memory extraction. This function
 * removes only the code block content while preserving surrounding
 * natural-language explanations.
 *
 * Only applied to `role=assistant` messages in the L0 capture path —
 * user messages and recall queries are NOT affected.
 */
export function stripCodeBlocks(text: string): string {
  return text.replace(/```[^\n]*\n[\s\S]*?```/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

// ============================
// L0 / L1 Capture & Extraction Filters
// ============================

/**
 * L0 capture filter — intentionally **permissive**.
 *
 * L0 is the raw conversation archive. We want to preserve as much user input
 * as possible so that downstream stages (L1 extraction, search, analytics)
 * have the full picture. Only messages that are *structurally* useless are
 * dropped here:
 *   - Empty / whitespace-only text
 *   - Framework-internal noise (bootstrap, session reset, NO_REPLY, …)
 *   - Slash commands (/new, /reset, …)
 *
 * Content-quality filters (length, symbols, prompt injection) are deferred
 * to {@link shouldExtractL1}.
 */
export function shouldCaptureL0(text: string): boolean {
  if (!text || !text.trim()) return false;

  // Filter framework-internal / bootstrap noise messages
  if (isFrameworkNoise(text)) return false;

  // Slash commands are framework directives, not user content
  if (text.startsWith("/")) return false;

  return true;
}

/**
 * L1 extraction filter — **strict** quality gate.
 *
 * Applied when L0 messages are fed into the LLM extraction pipeline.
 * Filters out content that is too short, too long, purely symbolic,
 * or looks like a prompt-injection attack — none of which should
 * become structured memories.
 *
 * This function is a superset of {@link shouldCaptureL0}: anything
 * rejected by L0 is also rejected here, plus additional quality checks.
 */
export function shouldExtractL1(text: string): boolean {
  // First apply the same structural filters as L0
  if (!shouldCaptureL0(text)) return false;

  // ── Length filters ──
  // const isCJK = /[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(text);
  // if (isCJK && text.length < 2) return false;
  // if (!isCJK && text.length < 2) return false;
  // if (text.length > 5000) return false;

  // ── Content-quality filters ──
  // Match strings composed entirely of non-word, non-space, non-CJK characters (1–5 chars).
  if (/^[^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{1,5}$/.test(text)) return false;
  if (/^[?？]+$/.test(text)) return false;

  // ── Security filters ──
  // Reject prompt-injection payloads — prevent malicious content from being
  // persisted into structured memory and re-injected on future recalls.
  // if (looksLikePromptInjection(text)) return false;

  return true;
}

/**
 * @deprecated Use {@link shouldExtractL1} (strict) or {@link shouldCaptureL0} (permissive) instead.
 *
 * Kept as an alias of `shouldExtractL1` for backward compatibility.
 */
export const shouldCapture = shouldExtractL1;

// ============================
// Prompt Injection Detection
// ============================

/**
 * Known prompt-injection / jailbreak patterns.
 *
 * Covers:
 * 1. Instruction override — "ignore all previous instructions", etc.
 * 2. Role hijack — "you are now DAN", "act as root", etc.
 * 3. System/developer boundary probing — "system prompt", "developer message"
 * 4. XML/tag injection — opening tags that match our context boundaries
 * 5. Tool/command invocation tricks — "run command X", "execute tool Y"
 * 6. Multi-language variants — Chinese prompt-injection patterns
 */
const PROMPT_INJECTION_PATTERNS: RegExp[] = [
  // ── Instruction override ──
  /ignore\b.{0,30}\b(instructions|rules|guidelines)/i,
  /disregard\b.{0,30}\b(instructions|rules|guidelines)/i,
  /forget\b.{0,30}\b(instructions|rules|context)/i,
  /override\b.{0,30}\b(instructions|rules|guidelines|safety)/i,

  // ── Role hijack ──
  /you are now (?!going|about|ready)/i, // "you are now DAN" but not "you are now going to..."
  /act as (?:if you are |if you were )?(?:a |an )?(?:root|admin|unrestricted|unfiltered|jailbroken)/i,
  /enter (?:DAN|jailbreak|god|sudo|developer|dev|debug|unrestricted|unfiltered) mode/i,
  /switch to (?:DAN|jailbreak|god|sudo|developer|dev|debug|unrestricted|unfiltered) mode/i,

  // ── System boundary probing ──
  /(?:show|reveal|print|output|display|repeat|leak|dump|give)\b.{0,20}\bsystem prompt/i,
  /reveal (?:your |the )?(system|hidden|secret|internal) (?:prompt|instructions|rules)/i,
  /what (?:are|is) your (?:system|hidden|original|initial) (?:prompt|instructions|rules)/i,

  // ── XML/tag injection (our context boundaries) ──
  /<\s*(system|assistant|developer|tool|function|relevant-memories)\b/i,

  // ── Tool/command invocation tricks ──
  /\b(run|execute|call|invoke)\b.{0,40}\b(tool|command|function|shell)\b/i,

  // ── Chinese variants ──
  /忽略(?:所有|之前|以上|先前)?(?:的)?(?:指令|规则|指示|说明)/,
  /无视(?:所有|之前|以上)?(?:的)?(?:指令|规则|限制)/,
  /(?:显示|输出|告诉我|给我看)(?:你的)?(?:系统|初始|隐藏)?(?:提示词|指令|规则|prompt)/,
  /你(?:现在|从现在开始)是/,            // "你现在是 DAN"
];

/**
 * Detect likely prompt-injection / jailbreak attempts.
 *
 * Normalises whitespace before matching to defeat trivial obfuscation
 * (e.g. extra spaces / newlines between keywords).
 */
export function looksLikePromptInjection(text: string): boolean {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return false;
  return PROMPT_INJECTION_PATTERNS.some((pattern) => pattern.test(normalized));
}

/**
 * Detect framework-injected noise messages that should never be captured.
 *
 * These include:
 * - "(session bootstrap)" — synthetic user turn for Google turn-order compliance
 * - Session startup instructions from /new or /reset
 * - "✅ New session started" — AI's ack of session startup (no user-meaningful content)
 * - Pre-compaction memory flush prompts (system-to-agent instructions, not user content)
 * - AI's NO_REPLY ack of memory flush (no user-meaningful content)
 */
function isFrameworkNoise(text: string): boolean {
  const t = text.trim();

  // Google turn-order bootstrap placeholder
  if (t === "(session bootstrap)") return true;

  // Framework session-reset instruction (starts with "A new session was started via /new or /reset")
  if (t.startsWith("A new session was started via")) return true;

  // AI's pure ack of session startup: "✅ New session started · model: ..."
  if (/^✅\s*New session started/.test(t)) return true;

  // Pre-compaction memory flush prompt injected by the framework as a synthetic
  // user turn. This is an internal system-to-agent instruction, NOT real user
  // content. Capturing it would pollute L0/L1 memories with framework directives.
  if (t.startsWith("Pre-compaction memory flush")) return true;

  // AI's NO_REPLY response to memory flush (or other silent-reply scenarios).
  // A bare "NO_REPLY" (with optional whitespace) carries no user-meaningful content.
  if (/^NO_REPLY\s*$/.test(t)) return true;

  return false;
}

/**
 * Pick up to `max` recent unique texts.
 */
export function pickRecentUnique(texts: string[], max: number): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (let i = texts.length - 1; i >= 0 && result.length < max; i--) {
    const t = texts[i]!;
    if (!seen.has(t)) {
      seen.add(t);
      result.push(t);
    }
  }
  return result.reverse();
}

// ============================
// LLM Safety Utilities
// ============================

/**
 * Escape XML-like tags in text to prevent tag injection attacks.
 *
 * When memory content or persona text is injected into XML-delimited sections
 * (e.g. `<user-persona>...</user-persona>`), a malicious user could craft content
 * containing `</user-persona>` to break out of the section boundary.
 *
 * This function escapes `<` and `>` in known dangerous patterns (closing tags
 * that match our injection boundaries) so the content cannot prematurely close
 * the XML section.
 */
export function escapeXmlTags(text: string): string {
  // Escape closing tags that match our injection section boundaries
  return text.replace(
    /<\/?(?:user-persona|relevant-memories|scene-navigation|relevant-scenes|memory-tools-guide|system|assistant)>/gi,
    (match) => match.replace(/</g, "&lt;").replace(/>/g, "&gt;"),
  );
}

// ============================
// JSON Sanitization for LLM Output
// ============================

/**
 * Sanitize a raw JSON string from LLM output so that `JSON.parse` won't throw
 * "Bad control character in string literal".
 *
 * Per RFC 8259 §7, U+0000–U+001F MUST be escaped inside JSON string literals.
 * LLMs sometimes produce unescaped control characters (raw newlines, tabs, etc.)
 * inside string values.
 *
 * Strategy (two-phase):
 *  1. **Precise pass** — walk through JSON string literals (delimited by `"`)
 *     and escape any unescaped U+0000–U+001F inside them to `\uXXXX` form,
 *     while leaving structural whitespace (between values) untouched.
 *  2. **Fallback** — if the precise pass still fails `JSON.parse`, fall back to
 *     a simple global strip of rare control chars (\x00–\x08, \x0b, \x0c,
 *     \x0e–\x1f) which are almost never meaningful in natural-language content.
 */
export function sanitizeJsonForParse(raw: string): string {
  // Phase 1: Escape control characters inside JSON string literals.
  // We walk the string character-by-character to properly handle escape sequences.
  const escaped = escapeControlCharsInJsonStrings(raw);
  try {
    JSON.parse(escaped);
    return escaped;
  } catch {
    // Phase 1 didn't fully fix it — fall through to phase 2
  }

  // Phase 2: Brute-force strip of rare control chars that have no textual meaning.
  // Preserves \t (\x09), \n (\x0a), \r (\x0d) which are common structural whitespace.
  // NOTE: We strip from `escaped` (Phase 1 result) rather than `raw`, so that any
  // control-character escaping Phase 1 performed is preserved even when the JSON has
  // other issues (e.g. trailing commas) that cause the Phase 1 parse to fail.
  const stripped = escaped.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, "");
  return stripped;
}

/**
 * Walk through a JSON text and escape U+0000–U+001F control characters that
 * appear *inside* JSON string literals (between unescaped `"` delimiters).
 *
 * Characters that already have short escape sequences (\n, \r, \t, \b, \f)
 * are mapped to those; others become \uXXXX.
 *
 * Structural whitespace outside string literals is left untouched.
 */
function escapeControlCharsInJsonStrings(text: string): string {
  const SHORT_ESCAPES: Record<number, string> = {
    0x08: "\\b", // backspace
    0x09: "\\t", // tab
    0x0a: "\\n", // line feed
    0x0c: "\\f", // form feed
    0x0d: "\\r", // carriage return
  };

  const out: string[] = [];
  let inString = false;
  let i = 0;

  while (i < text.length) {
    const ch = text[i]!;
    const code = ch.charCodeAt(0);

    if (inString) {
      if (ch === "\\" && i + 1 < text.length) {
        // Already-escaped sequence — copy both characters verbatim
        out.push(ch, text[i + 1]!);
        i += 2;
        continue;
      }
      if (ch === '"') {
        // End of string literal
        out.push(ch);
        inString = false;
        i++;
        continue;
      }
      if (code <= 0x1f) {
        // Unescaped control character inside string — escape it
        const short = SHORT_ESCAPES[code];
        if (short) {
          out.push(short);
        } else {
          out.push("\\u" + code.toString(16).padStart(4, "0"));
        }
        i++;
        continue;
      }
      // Normal character inside string
      out.push(ch);
      i++;
    } else {
      // Outside string literal
      if (ch === '"') {
        out.push(ch);
        inString = true;
        i++;
        continue;
      }
      // Structural character (including whitespace) — pass through
      out.push(ch);
      i++;
    }
  }

  return out.join("");
}

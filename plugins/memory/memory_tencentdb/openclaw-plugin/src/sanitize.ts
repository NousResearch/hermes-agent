/**
 * Text sanitization for L0 capture (client-side).
 * Mirrors the cleaning logic in extensions/memory-tencentdb/src/utils/sanitize.ts
 * — kept in sync to ensure the client and server filter the same noise.
 */

/** Strip injected memory tags + framework metadata blocks + media markers. */
export function sanitizeText(text: string): string {
  let cleaned = text;

  // Remove injected memory context tags (prevent feedback loops on re-capture)
  cleaned = cleaned.replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/g, "");
  cleaned = cleaned.replace(/<user-persona>[\s\S]*?<\/user-persona>/g, "");
  cleaned = cleaned.replace(/<relevant-scenes>[\s\S]*?<\/relevant-scenes>/g, "");
  cleaned = cleaned.replace(/<scene-navigation>[\s\S]*?<\/scene-navigation>/g, "");
  cleaned = cleaned.replace(/<memory-tools-guide>[\s\S]*?<\/memory-tools-guide>/g, "");

  // Offload-injected task context blocks
  cleaned = cleaned.replace(/<current_task_context>[\s\S]*?<\/current_task_context>/g, "");
  cleaned = cleaned.replace(/<history_task_context[\s\S]*?<\/history_task_context>/g, "");

  // Framework-injected inbound metadata blocks (label + ```json ... ```)
  cleaned = cleaned.replace(
    /(?:Conversation info|Sender|Thread starter|Replied message|Forwarded message context|Chat history since last reply)\s*\(untrusted[\s\S]*?\):\s*```json\s*[\s\S]*?```/g,
    "",
  );

  // Legacy conversation metadata JSON blocks
  cleaned = cleaned.replace(/```json\s*\{[\s\S]*?"session[\s\S]*?\}\s*```/g, "");

  // Reply directive tags: [[reply_to_current]]
  cleaned = cleaned.replace(/\[\[reply_to[^\]]*\]\]\s*/g, "");

  // Skill-selection wrappers: ¥¥[ ... ]¥¥
  cleaned = cleaned.replace(/¥¥\[[\s\S]*?\]¥¥/g, "");

  // Line-leading timestamps: [Tue 2026-03-24 03:48 UTC] / GMT+8 / GMT+5:30
  cleaned = cleaned.replace(/^\[[\w\d\-:+ ]+\]\s*/gm, "");

  // Gateway media-attachment markers
  cleaned = cleaned.replace(/\[media attached:[^\]]*\]\s*/g, "");

  // Gateway image-reply instructions
  cleaned = cleaned.replace(
    /To send an image back,[\s\S]*?(?:Keep caption in the text body\.)\s*/g,
    "",
  );

  // System exec blocks: "System: [timestamp] Exec completed ..."
  cleaned = cleaned.replace(/^System:\s*\[[\s\S]*?$/gm, "");

  // Inline base64 image data URIs
  cleaned = cleaned.replace(/data:image\/[a-z+]+;base64,[A-Za-z0-9+/=]+/gi, "");

  // Null chars + collapse whitespace
  cleaned = cleaned.replace(/\0/g, "").replace(/\n{3,}/g, "\n\n").trim();

  return cleaned;
}

/**
 * Strip fenced code blocks from assistant replies before L0 capture.
 * Only applied to role=assistant — keeps explanatory text but drops noisy code.
 */
export function stripCodeBlocks(text: string): string {
  return text.replace(/```[^\n]*\n[\s\S]*?```/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * L0 capture filter — permissive. Only drops messages that are structurally
 * useless (empty, framework bootstrap noise, slash commands).
 */
export function shouldCaptureL0(text: string): boolean {
  if (!text || !text.trim()) return false;
  if (isFrameworkNoise(text)) return false;
  if (text.startsWith("/")) return false;
  return true;
}

function isFrameworkNoise(text: string): boolean {
  const t = text.trim();
  if (t === "(session bootstrap)") return true;
  if (t.startsWith("A new session was started via")) return true;
  if (/^✅\s*New session started/.test(t)) return true;
  if (t.startsWith("Pre-compaction memory flush")) return true;
  if (/^NO_REPLY\s*$/.test(t)) return true;
  return false;
}

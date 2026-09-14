/** Whole-message garbage that must not paint as an assistant bubble.

Observed 2026-09-14 on Codex / Muse Spark / gpt-5.6-terra tool-carrying turns:
a bare UUID, vim `?wq`, mixed-script fragments, ZWSP-spliced tokens. These were
persisted as `content` and shown instead of the real reply. Latin and Hebrew
short replies stay visible. */

const UUID_ONLY_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const VIM_EX_RE = /^(?:\?wq|:wq!?|:q!?)$/
const ZWSP = '\u200b'

function nonLatinScriptBuckets(text: string): Set<string> {
  const buckets = new Set<string>()

  for (const char of text) {
    const code = char.codePointAt(0)

    if (code === undefined) {
      continue
    }

    if (code >= 0x0400 && code <= 0x04ff) {
      buckets.add('cyrillic')
    } else if (code >= 0x0370 && code <= 0x03ff) {
      buckets.add('greek')
    } else if ((code >= 0x0600 && code <= 0x06ff) || (code >= 0x0750 && code <= 0x077f)) {
      buckets.add('arabic')
    } else if (code >= 0x0900 && code <= 0x097f) {
      buckets.add('devanagari')
    } else if ((code >= 0x1100 && code <= 0x11ff) || (code >= 0xac00 && code <= 0xd7af)) {
      buckets.add('hangul')
    } else if (code >= 0x1780 && code <= 0x17ff) {
      buckets.add('khmer')
    } else if ((code >= 0x3040 && code <= 0x30ff) || (code >= 0x31f0 && code <= 0x31ff)) {
      buckets.add('japanese')
    } else if (code >= 0x4e00 && code <= 0x9fff) {
      buckets.add('han')
    }
  }

  return buckets
}

export function isDegenerateAssistantText(text: string): boolean {
  const stripped = text.trim()

  if (!stripped) {
    return false
  }

  if (UUID_ONLY_RE.test(stripped) || VIM_EX_RE.test(stripped)) {
    return true
  }

  if (stripped.includes(ZWSP) && stripped.length <= 40) {
    return true
  }

  return stripped.length <= 24 && nonLatinScriptBuckets(stripped).size >= 2
}

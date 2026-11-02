// JSON.stringify emits U+2028, U+2029 and U+0085 raw inside strings. Readers
// with splitlines() semantics (httpx LineDecoder, Python str.splitlines) treat
// those code points as line breaks, so an NDJSON frame containing them is
// silently split into fragments that fail JSON parsing on the consumer side.
// Escaping them to \\uXXXX keeps every frame a single \\n-terminated line while
// staying spec-valid JSON: JSON.parse() decodes the original characters back.
const LINE_SEPARATORS = new RegExp(
  "[" + String.fromCharCode(0x2028, 0x2029, 0x0085) + "]",
  "g"
);

export function escapeLineSeparators(line) {
  return line.replace(LINE_SEPARATORS, (c) =>
    "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0")
  );
}

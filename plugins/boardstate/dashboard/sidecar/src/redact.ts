// Secret redaction for agent-facing error messages (invariant #3 / SEC-2 / SEC-3).
//
// Build a redactor from the exact secret strings that must never surface to the agent
// (connector command/url/args + env keys/values + header values, plus the sidecar nonce and
// operator secret). Two properties matter:
//   * length-agnostic for config-sourced secrets (SEC-2): an API key can be short — a minimum
//     length filter would let it leak. (Callers apply a length floor only to the long random
//     nonce/secret, to avoid over-redacting unrelated text.)
//   * DESCENDING length order (SEC-3): replace longer secrets first, so a shorter secret that
//     is a PREFIX of a longer token-bearing value can't mask only the prefix and leave the
//     token suffix visible (`[redacted]-secret…`).

/** Build a redactor that replaces every occurrence of each secret with `[redacted]`. */
export function buildRedactor(secrets: readonly string[]): (message: string) => string {
  const sorted = [...new Set(secrets)]
    .filter((s) => typeof s === "string" && s.length >= 1)
    .sort((a, b) => b.length - a.length);
  if (sorted.length === 0) {
    return (message) => message;
  }
  return (message) => {
    let out = message;
    for (const secret of sorted) {
      out = out.split(secret).join("[redacted]");
    }
    return out;
  };
}

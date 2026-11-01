# Redaction heuristic patterns

The `council.py` safety-net redaction pass runs seven regex
patterns against the post-Ollama text. Any hit fails-closed
(exit 4, raw artifact + report preserved in the tempdir).

## The seven patterns

| Name | Pattern | Catches |
|---|---|---|
| `email` | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` | Standard email addresses (RFC 5322 lite) |
| `phone` | `\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b` | NANP phone numbers, with optional `+1` and various separators |
| `ssn` | `\b\d{3}-\d{2}-\d{4}\b` | US Social Security Numbers (the literal `XXX-XX-XXXX` shape) |
| `openai_sk` | `\bsk-[A-Za-z0-9]{20,}\b` | OpenAI `sk-…` secret keys (20+ trailing chars) |
| `github_pat` | `\bghp_[A-Za-z0-9]{20,}\b` | GitHub personal access tokens (fine-grained + classic PAT prefix) |
| `slack` | `\bxox[baprs]-[A-Za-z0-9-]{10,}\b` | Slack bot, app, user, refresh, or legacy tokens |
| `aws_akia` | `\bAKIA[0-9A-Z]{16}\b` | AWS IAM access key IDs (root + IAM user) |

These are intentionally **narrow**. False positives (catching a
test artifact that *looks* like a secret) are preferred over
false negatives (missing a real one), but the patterns are
specific enough that a markdown plan or a YAML config with no
secrets will pass cleanly.

## What these patterns WILL miss (and why `--forbid-regex` exists)

- **Custom internal tokens.** Your company's internal API keys
  don't match any of these patterns. Use `--forbid-regex
  "<your-pattern>"` to add a project-specific check that runs
  *after* the seven built-ins.
- **AWS secret access keys** (the `aws_secret_access_key =
  "wJalr..."` value, not the `AKIA…` access key ID). They don't
  match any built-in pattern because they have no fixed prefix.
  Add `--forbid-regex` if your artifacts touch AWS.
- **Bearer tokens in headers** (`Authorization: Bearer
  eyJhbGciOi...`). Use `--forbid-regex` with
  `Bearer\s+[A-Za-z0-9._-]{20,}`.
- **IPv4 + port + basic-auth URLs** (`http://user:pass@1.2.3.4:80`).
  Use `--forbid-regex` with `://[^\s/]+:[^\s@]+@`.
- **Credit card numbers** (PAN). Not in the built-ins because
  they're not a secret per se, but they often shouldn't leave
  the box. Use `--forbid-regex` with the Luhn-tested shape.

## What these patterns WILL false-positive on (and why that's OK)

- **Test fixtures that look like real secrets.** A markdown
  document with `ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
  will trip the GitHub PAT pattern, even if it's clearly a
  placeholder. That's the design — council.py errs on the
  side of refusing to send. If you genuinely want to send
  such an artifact, redact the placeholder yourself before
  piping.
- **Phone numbers in legitimate non-PII contexts.** A document
  saying "call our support line at 555-123-4567" will trip the
  phone pattern, even if `555` numbers aren't real. Same
  trade-off; redact manually if you know the document is
  clean.

## Test data discipline

In selftest on 2026-06-16, fake test data like `sk-123...mnop`
or `AKIAIO...MPLE` failed to trip the heuristic because the
regex requires `AKIA` followed by exactly 16 chars of
`[0-9A-Z]` (no literal dots). When testing the heuristic
directly, use **properly shaped** fake data:

```text
AKIAIOSFODNN7EXAMPLE    # 16 trailing chars
ghp_aBcDeFgHiJkLmNoPqRsT   # 20+ trailing chars
sk-1234567890abcdefghijklmnop   # 20+ trailing chars
```

(These are the AWS / GitHub / OpenAI example values from their
respective docs — well-known, not real production secrets, safe
to use as test fixtures.)

If a heuristic pass returns `hits: []` on data you believe is
dirty, first check whether the data is **shaped like** a real
secret. A `...` in the middle of a token is a tell that the
test data is malformed.

## When to extend the built-ins

Add a new built-in pattern (i.e. promote something from
`--forbid-regex` to a permanent fixture) when:

- You find yourself passing the same `--forbid-regex` in
  three or more unrelated sessions.
- The pattern catches a class of secret that's high-impact and
  high-frequency in your real artifacts (e.g. internal
  customer IDs, financial account numbers).
- You're confident the false-positive rate stays low — i.e. the
  pattern is specific enough that legitimate markdown / YAML /
  code artifacts don't trip it.

For everything else, `--forbid-regex` is the right answer. It
keeps the built-ins small and the failure mode understandable.

# Security Review Checklist

Use this checklist alongside the `code-review-and-quality` skill when reviewing a change for security regressions. Every item below must be answered before a security-sensitive change merges.

## Input & Boundaries
- [ ] All external input (HTTP params, headers, body, file uploads, CLI args, env, config files) is validated and sanitized at the system boundary
- [ ] Untrusted data is treated as untrusted everywhere — never trust values from APIs, logs, user content, or config unless re-validated
- [ ] Type/shape/range checks reject invalid input instead of coercing it

## Secrets & Credentials
- [ ] No secrets, tokens, passwords, API keys, or private keys in source, logs, or version control
- [ ] `.env`, `*.key`, `*.pem`, and similar are git-ignored
- [ ] Logs redact secrets and full PII (allowlist fields; never log whole request bodies)

## Authentication & Authorization
- [ ] Auth checks run on every protected path (not just the happy path)
- [ ] Authorization is enforced server-side, not only in the UI
- [ ] Privilege changes (role, scope, tenant) require explicit verification

## Injection & Data Safety
- [ ] Database queries use parameterized statements / ORM bindings — no string concatenation of SQL
- [ ] Shell commands avoid unsanitized interpolation; prefer argument arrays
- [ ] HTML/JS output is encoded to prevent XSS
- [ ] Path handling blocks traversal (`../`) and enforces allowed roots
- [ ] XML/deserialization uses safe parsers (no external entity / unsafe type mapping)

## Dependencies
- [ ] New dependencies come from trusted sources with no known vulnerabilities (`npm audit`, `dotnet list package --vulnerable`, etc.)
- [ ] License of each dependency is compatible with the project
- [ ] Lockfile diff reviewed; no hand-edited lockfile

## Transport & Crypto
- [ ] TLS used for any network egress/ingress of sensitive data
- [ ] No disabled cert validation, no `verify=False`, no hardcoded crypto keys
- [ ] Token/session lifecycle follows the platform standard (short-lived, rotated, scoped)

If any item fails, mark it **Critical** and block merge until resolved.

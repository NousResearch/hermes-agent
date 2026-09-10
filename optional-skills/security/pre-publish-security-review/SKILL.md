---
name: pre-publish-security-review
description: Review code and artifacts safely before publishing.
version: 0.1.0
author: Mark S. (unsupportedpastels), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, code-review, publishing, owasp, secrets, sast]
    category: security
    related_skills: [github-code-review, publish-site, cloudflare-temporary-deploy]
---

# Pre-Publish Security Review Skill

Review the exact code and files about to become public before a GitHub push or
site deployment. This is a short, evidence-based safety gate for secret
exposure and common web risks; like Hermes' other in-process checks, it is a
review aid, not a security boundary. It does not replace OS-level isolation, a
full security assessment, penetration test, or project-specific SAST workflow.

## When to Use

- Before `publish-site`, `cloudflare-temporary-deploy`, a public repository
  push, or another action that makes code or build output externally reachable.
- After a meaningful change to authentication, authorization, database access,
  templates, command execution, URL fetching, dependencies, or error handling.
- When the user asks for a friendly security check rather than a full audit.

Do not use this as proof that an application is secure. For a full-repository
review, use a dedicated project security skill when one is available. For an
ordinary PR-only review with no pending publication, use `github-code-review`.
Use this skill when the PR's code or generated artifact is about to become
public, or upon the user's direct request.

## Prerequisites

- A local repository or publish directory.
- The `terminal`, `read_file`, and `search_files` tools.
- Python 3 for the bundled artifact inventory helper.
- Optional project-declared tests and already-installed scanners. Do not install
  or upload source to a third-party scanner without approval.

The OWASP mapping below follows the
[OWASP Top 10:2025](https://owasp.org/Top10/2025/). The list is an awareness
baseline, not an exhaustive vulnerability taxonomy.

## How to Run

Use `terminal` from the project root. First identify both review surfaces:

1. **Code surface:** the staged diff or branch diff that will be pushed.
2. **Publish surface:** the exact directory a hosting command will upload, such
   as `dist/`, `build/`, or the repository root.

Inventory the publish surface without printing matched secret values:

```text
python "${HERMES_SKILL_DIR}/scripts/inventory_publish_surface.py" <publish-dir>
```

Exit `1` means the helper found a blocking sensitive path or high-confidence
credential signature. Exit `0` means no deterministic blocker was found; it is
not a clean bill of health. Then inspect the code diff and apply the review
lenses below.

## Quick Reference

| Task | Command or evidence |
| --- | --- |
| Inventory artifact | `python "${HERMES_SKILL_DIR}/scripts/inventory_publish_surface.py" <publish-dir>` |
| Review staged code | `git diff --cached --stat` then `git diff --cached` through `terminal` |
| Review branch code | `git diff <base>...HEAD --stat` then the full diff through `terminal` |
| Trace a finding | `read_file` around the changed source-to-sink path |
| Run project checks | Repository-declared tests/scanners through `terminal` |
| Set the outcome | Confirmed evidence + residual risk → `Ready to publish`, `Fix before publishing`, or `Need your input` |

Use `Fix before publishing` for a secret or private key in the publish surface,
a confirmed reachable high-impact vulnerability, a failing affected security
test, or an unresolved injection/access-control path. Use `Need your input` when
a safe fix requires the user's choice or a coverage gap cannot be closed. Scanner
absence alone is a coverage warning, not a vulnerability.

## Review Lenses

### Publish-surface exposure

- Treat every file in a static bundle as public, including source maps, comments,
  JSON, client configuration, and files hidden by a leading dot.
- A `.gitignore` entry is not evidence that a deploy CLI excludes the file.
  Review the actual upload directory.
- Browser-delivered environment values are public by design. Server-side secrets
  belong in provider secret storage, never in source or generated JavaScript.
- Do not paste suspected values into chat or reports. Record only path, line,
  rule label, and remediation.

### OWASP Top 10:2025 changed-code pass

Review categories that intersect the diff. Mark a category `N/A` only with a
short reason; do not imply whole-repository coverage.

| Category | Focus on changed code |
| --- | --- |
| A01 Broken Access Control | Ownership/tenant checks, role gates, IDOR, deny-by-default server routes |
| A02 Security Misconfiguration | Debug modes, permissive CORS/CSP, public admin paths, source maps, unsafe defaults, and the real public HTTPS path rather than local HTTP alone |
| A03 Software Supply Chain Failures | New dependencies, lockfile drift, unpinned actions, install/build scripts |
| A04 Cryptographic Failures | Hardcoded keys, weak randomness/hashing, custom crypto, sensitive transport/storage |
| A05 Injection | Parameterized SQL, argument-safe process calls, output encoding, template/DOM sinks, path input |
| A06 Insecure Design | Missing abuse limits, trust-boundary assumptions, unsafe business workflow |
| A07 Authentication Failures | Session lifecycle, account recovery, credential checks, brute-force controls |
| A08 Software or Data Integrity Failures | Unsafe deserialization, unsigned updates, untrusted build or webhook data |
| A09 Security Logging and Alerting Failures | Security-event coverage without tokens, passwords, or personal data in logs |
| A10 Mishandling of Exceptional Conditions | Fail-open authorization, partial writes, insecure fallback, leaked stack traces |

#### HTTPS and edge termination (A02)

Judge transport security from the deployment users actually have. Choose the
smallest matching path instead of assuming a server backend:

- **Managed static site or SPA (the common path):** when `publish-site` sends a
  static output directory to GitHub Pages, Cloudflare Pages, or Netlify, the
  provider owns the public origin and HTTPS. Check the final HTTPS URL,
  HTTP-to-HTTPS redirect, mixed content, and accidental secret files. There is
  no application server origin, proxy trust, ops cookie, or backend port to
  review; do not invent those findings.
- **Server-rendered app or API:** use the origin/proxy checks below only when the
  project actually runs a Node/Python/Ruby/etc. server, exposes an API, or keeps
  a long-running container behind the HTTPS provider.

For either path:

- `http://localhost` and other local preview URLs are expected and are not a
  finding by themselves.
- The public/share URL must use HTTPS. When the provider exposes a public HTTP
  URL, verify that it redirects to the same host over HTTPS.
- Flag browser-visible `http://` assets, API URLs, form actions, or `ws://`
  sockets in production output. They create mixed content or bypass the HTTPS
  page; use relative URLs, HTTPS, or WSS instead.

For a server runtime only:

- An origin may use HTTP when a trusted platform owns or isolates that hop and
  terminates HTTPS at the edge. Do not tell the user to add application-level
  TLS solely to pass this review.
- Flag a directly reachable HTTP origin that bypasses the intended edge,
  authentication, rate limits, or headers. If edge-to-origin traffic crosses an
  untrusted network, require HTTPS on that hop rather than treating edge TLS as
  end-to-end protection.
- Confirm that forwarded scheme handling is trusted correctly and session
  cookies are still marked `Secure`.

If a live URL already exists, use `terminal` to verify the HTTPS response and
HTTP-to-HTTPS redirect. Keep the user-facing result simple: a managed static
site with provider HTTPS, or local HTTP behind a verified HTTPS server host,
needs no transport fix; public HTTP or mixed content does.

Also inspect changed outbound URL fetching for SSRF and redirect abuse even
though SSRF is not a standalone 2025 category.

## Procedure

1. **Resolve and classify the external action.** Start with the common static
   path: if the host receives only an output directory containing HTML/CSS/JS,
   use the managed-static checks and do not inspect imaginary server controls.
   Switch to the server-runtime path only when the project actually runs a
   server/API/container after deployment. Record the target, branch/commit, and
   exact upload directory or image. Stop if the publish surface is unknown.
   Done when every file that can become public belongs to a named surface and
   only relevant security controls are in scope.

2. **Inspect repository instructions and status.** Use `read_file` for relevant
   `AGENTS.md`, `CONTRIBUTING.md`, and security guidance; use `terminal` for
   branch, status, and diff metadata. Do not discard unrelated local changes.
   Done when the base revision and working-tree ownership are clear.

3. **Inventory the publish surface.** Run the bundled helper against the exact
   upload directory. Read its JSON labels, never secret values. Treat
   `sensitive_paths` and `secret_candidates` as blockers until inspected and
   removed or proven to be inert test/example data outside the upload surface.
   Manually disposition every `review_candidates` entry and every
   `skipped_files` entry before choosing `Ready to publish`; they are coverage
   gaps, not passes. Done when blockers are cleared, review gaps have
   dispositions, and the corrected surface has been rescanned.

4. **Build a changed-code map.** Review the complete staged or branch diff, then
   use `read_file` around changed security-sensitive functions. Group changes
   by input, authorization, data store, template/DOM output, process execution,
   outbound network, dependencies/build, logging, and error handling. Done when
   each changed trust boundary has an owner and data flow.

5. **Run the applicable OWASP pass.** For each intersecting category, trace a
   plausible untrusted input to its security-sensitive sink and verify the
   control in source. For SQL, prove parameter binding at the final query call;
   ORM use alone is not proof. For shell calls, prove arguments do not cross an
   unsafe shell-string boundary. Done when each applicable category has source
   evidence, a confirmed finding, or a meaningful negative result.

6. **Run project-native checks.** Prefer repository-declared tests, linters,
   dependency audits, and existing scanners through `terminal`. Keep scanner
   installs isolated and ask before adding dependencies or sending code to an
   external service. A scanner match is a lead until source review confirms
   reachability and impact. Done when each selected command has its actual
   result and each unavailable check is recorded as a coverage gap.

7. **Choose a plain-language outcome.** Use `Fix before publishing` when a
   confirmed problem remains. Use `Need your input` when the agent needs the
   user to rotate a key, choose a provider setting, approve a dependency change,
   or accept a coverage gap. Use `Ready to publish` only when the selected
   surfaces and affected security paths are verified. Never publish merely
   because a scanner returned zero findings. Done when the outcome follows from
   evidence and the next action is obvious to a non-security specialist.

8. **Fix what the agent safely can.** When remediation is already authorized,
   use `patch` or `write_file` for narrow local fixes, run affected tests,
   rebuild the artifact, and repeat the inventory. Group similar issues under
   one fix instead of producing a long findings catalog. Ask the user only for
   actions the agent cannot safely complete, such as rotating a real credential
   or choosing whether a public feature should remain exposed. Done when the
   final reviewed artifact is the unchanged artifact sent to the provider.

## Pitfalls

- Reviewing source while deploying stale or separately generated output.
- Scanning while a build or watcher can replace files. Stop concurrent writers,
  rebuild once, scan, and publish that unchanged artifact.
- Checking only tracked files; `.env.production` may be untracked yet still sit
  inside `dist/`.
- Printing the matched token as evidence. A rule label and location are enough.
- Treating minified code, an ORM, escaping middleware, or a WAF as automatic
  proof that injection is impossible.
- Reporting every regex or scanner match as a vulnerability without a reachable
  source-to-sink path.
- Claiming all OWASP categories were reviewed when the work covered only a diff.
- Blocking publication only because an optional scanner is unavailable; report
  the coverage gap and increase manual review instead.
- Running active probes against a live target without separate authorization.

## Verification

Keep the default result short and practical. Do not lead with severity tables,
OWASP category names, scanner rule IDs, raw command output, or a long findings
index. Keep that evidence in working notes and show it only when the user asks
or when it is needed to explain a fix.

Use one of these outcomes exactly:

- **Ready to publish** — no obvious blocker remains in the reviewed surfaces.
- **Fix before publishing** — the agent found a confirmed problem it can explain
  and fix or hand back clearly.
- **Need your input** — a safe next step requires a user decision or action.

Group findings that share one fix. Include only confirmed items that change what
happens next. Each item needs a file or component, one plain-language sentence
about the problem, and the exact fix. Never include a matched secret value.

Use this default shape, omitting empty sections:

```markdown
## Publish safety check

**Result:** Ready to publish | Fix before publishing | Need your input

<One or two sentences explaining the result in everyday language.>

### Fixes I can make
- `<path or component>` — <what is wrong>. I will <specific fix and check>.

### What you need to do
- <one concrete user step, such as rotate a key or choose a visibility setting>

### Next step
- <what the agent will do after the fix or decision>
```

Examples of useful wording:

- `dist/.env.production` would become public. Remove it from the upload, move
  real values to provider secret storage, rebuild, and scan again.
- `server/users.ts:84` builds SQL from user input. Change it to a parameterized
  query and add a regression test before publishing.
- A real key may have been exposed. The user must revoke or rotate it; deleting
  the file alone does not make the old key safe.
- The app uses HTTP only for local preview, while the verified public URL uses
  provider-managed HTTPS. No transport fix is needed.
- The production bundle calls `http://api.example.com`. Change it to HTTPS or a
  relative URL so the browser does not block or downgrade the request.

When the outcome is `Ready to publish`, say that no obvious blockers were found
in the reviewed code and artifact, that this is not a guarantee, and whether the
user has any remaining step. Before returning that outcome, verify that the
inventory helper exits `0`, every review/skipped item has a disposition, the
artifact was rebuilt after any fix, affected tests pass, and the reported paths
still match the exact artifact that will be published.

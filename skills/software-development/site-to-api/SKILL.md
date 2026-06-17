---
name: site-to-api
description: Use when the operator wants Hermes to turn observed local browser traffic into an OpenAPI spec, JavaScript client, and reusable API workflow without Browserbase cloud.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [browser, cdp, openapi, api-client, local-first, site-to-api]
    related_skills: [codex, dogfood, codebase-inspection]
---

# Site To API

## Overview

Use this skill to turn a website's observed HTTP traffic into a documented API surface. The preferred runtime is local-first: a Chrome DevTools Protocol capture runs on the operator's machine, then emits OpenAPI, a JavaScript client, and a short report.

This skill does not require Browserbase cloud, a Browserbase API key, or the `browse` CLI. If those are available for another workflow, treat them as optional alternatives, not prerequisites.

Expected outputs:

- `openapi.yaml`
- `openapi.json`
- `client.mjs`
- `report.md`
- `confidence.json`

## When To Use

Use this skill when the operator asks for:

- "turn this site into an API"
- "make an API/client/OpenAPI from this website"
- "use browser-to-api locally"
- "Browserbase 없이 Hermes/Codex가 쓰게 만들어"
- "내 로그인 세션으로 자동 실행해"

Do not use it for public deployment, destructive account actions, payments, credential rotation, customer messaging, or changing real production data unless the operator explicitly authorizes that specific side effect.

## Runtime Discovery

Start with the local wrapper when it exists:

```bash
/Users/yu/.hermes/scripts/site-to-api-local.sh doctor
```

If that wrapper is absent, look for `site-to-api` in `PATH`:

```bash
site-to-api --doctor
```

If neither command exists, stop and report that the runtime is missing. Do not ask for a Browserbase API key as a substitute for this local workflow.

Common local paths on the operator's machine:

| Path | Purpose |
| --- | --- |
| `/Users/yu/bin/site-to-api` | Local Chrome/CDP capture and generator entrypoint |
| `/Users/yu/bin/local-site-to-api.mjs` | Node runtime implementation |
| `/Users/yu/tools/site-to-api/generator` | Local OpenAPI/client generator |
| `/Users/yu/.site-to-api/sites.json` | Remembered site registry |
| `/Users/yu/.site-to-api/profiles/<slug>` | Stable local Chrome profiles |
| `/Users/yu/.o11y/<run-id>/api-spec/` | Generated OpenAPI/client/report artifacts |

## Standard Workflow

1. Check readiness:

   ```bash
   /Users/yu/.hermes/scripts/site-to-api-local.sh doctor
   ```

2. For a site that needs login or consent, prepare the stable local profile once:

   ```bash
   /Users/yu/.hermes/scripts/site-to-api-local.sh login https://example.com example --hold 180
   ```

   The operator can complete login in the local Chrome window. Later `run` calls reuse the same local profile.

3. Generate an API spec/client:

   ```bash
   /Users/yu/.hermes/scripts/site-to-api-local.sh run example example-api --origins example.com --title ExampleAPI --no-open
   ```

   A URL can be used directly when no registry slug exists:

   ```bash
   /Users/yu/.hermes/scripts/site-to-api-local.sh run https://example.com example-api --origins example.com --title ExampleAPI --no-open
   ```

4. Inspect artifacts:

   ```bash
   ls /Users/yu/.o11y/example-api/api-spec
   sed -n '1,160p' /Users/yu/.o11y/example-api/api-spec/report.md
   ```

5. If the result should become a reusable Codex skill:

   ```bash
   /Users/yu/bin/browser-to-api-site-init example-api example-api --display "Example API"
   ```

## Local Session Policy

This skill may reuse the operator's own local logged-in Chrome profile when the target site belongs to the operator or the operator is authorized to access it.

Rules:

- Use `login` once for human login or consent.
- Use `run` for repeated capture after that profile is ready.
- Store generated artifacts, not secrets.
- Do not print cookies, bearer tokens, one-time codes, private headers, or raw authorization values.
- If a site requires fresh human verification or access approval, record the blocker and stop that branch.

Blocker files usually appear at:

```bash
/Users/yu/.o11y/<run-id>/blocker.json
```

## Output Review

After every run, report these items:

```text
STATUS=<completed|blocked|failed>
RUN_ID=<run-id>
ARTIFACTS=/Users/yu/.o11y/<run-id>/api-spec
OPENAPI=<present|missing>
CLIENT=<present|missing>
REPORT=<present|missing>
BLOCKER=<none|/Users/yu/.o11y/<run-id>/blocker.json>
```

For API-client proof, prefer a small read-only call:

```bash
node --input-type=module -e "import * as api from '/Users/yu/.o11y/<run-id>/api-spec/client.mjs'; console.log(Object.keys(api));"
```

If the generated client contains an obvious safe GET helper, call that helper and print only a small non-sensitive shape of the result.

## Common Pitfalls

1. Confusing this with Browserbase. This skill is local-first and should not request a Browserbase API key.
2. Running non-interactive before login. First use `login` with a stable site profile, then use `run`.
3. Treating every captured endpoint as production-ready. Review `confidence.json` and `report.md`; low-confidence endpoints need manual confirmation.
4. Printing sensitive headers or cookies from captured traffic. Report redaction status and artifact paths instead.
5. Claiming automation succeeded when the site showed a verification or access blocker. Cite `blocker.json` and the next safe local step.

## Verification Checklist

- [ ] Local `doctor` passes through `/Users/yu/.hermes/scripts/site-to-api-local.sh doctor` or `site-to-api --doctor`.
- [ ] Run directory exists under `/Users/yu/.o11y/<run-id>/`.
- [ ] `api-spec/openapi.yaml`, `api-spec/openapi.json`, `api-spec/client.mjs`, and `api-spec/report.md` exist.
- [ ] `confidence.json` was reviewed for low-confidence routes.
- [ ] Any blocker is recorded as `blocker.json` and reported as blocked, not completed.
- [ ] No secrets, cookies, or private auth headers were printed in the final report.

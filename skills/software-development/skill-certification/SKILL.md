---
name: skill-certification
description: Certify and verify AI agent skills via SkillSeal.
version: 1.0.0
author: Workloop (Danilo Almeida) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [skills, certification, security, verification, agents]
    category: software-development
    requires_toolsets: [web, terminal]
---

# Skill Certification Skill

Submit an AI agent skill (a `SKILL.md` bundle) to SkillSeal for functional and security certification, or verify an existing certificate. This skill wraps the public SkillSeal API so an agent can get an authoritative, evidence-based verdict on whether a skill actually works and stays safe. It does not run the skill locally; SkillSeal does that in an isolated sandbox.

## When to Use

Use this skill when any of the following is true:

- the user wants to certify a skill they wrote or downloaded, and get a verifiable certificate;
- the user wants to check whether a certificate code (`SKL-XXXX-XXXX-XX`) is genuine before trusting a third-party skill;
- the user needs a security/functional verdict on a skill before installing it into an agent;
- the user wants to fetch the public SkillSeal certification report.

Do not use it for general skill *authoring* guidance (use `hermes-agent-skill-authoring` for that) or for running arbitrary untrusted code locally.

## Prerequisites

- Network access (SkillSeal is a public web API).
- `web_extract` or `terminal` with `curl` available.
- No API key required — the public endpoints are open.

## How to Run

### Verify a certificate (most common)

Given a code like `SKL-D266-F285-BC`, verify it:

```
curl "https://skillseal.workloop.com.br/api/verify?code=SKL-D266-F285-BC"
```

Interpret the response:

| Field | Meaning |
|---|---|
| `status: CERTIFIED` | Genuine, signed certificate — the skill was tested and passed |
| `status: INVALID` | Certificate failed verification (tampered / unknown issuer) |
| `status: NOT_VERIFIED` / 404 | No valid certificate for this code |

A `CERTIFIED` response also includes `skill`, `author`, `content_sha256`, and `checks` — the exact layers validated and (for new certificates) the tests that actually ran.

### Certify a skill

The skill must be a `.zip`, `.tar.gz`, or a single `.md`/`.py`/`.sh`/`.js`/`.sql` file. Submit it:

```
curl -F "name=YourName" -F "email=you@example.com" \
     -F "file=@./my-skill.zip" \
     "https://skillseal.workloop.com.br/submit"
```

The response is near-instant and returns a `veredito`:

| `veredito` | Meaning |
|---|---|
| `CERTIFICADO` | Passed — a `certificado` (SKL code) is included |
| `NAO_CERTIFICADO` | Failed (functional and/or security) — read `detalhe` |
| `DUPLICADO` | Content already certified (exact copy of an existing skill) |
| `ERRO_*` | Extraction or evaluation error |

If certified, relay the `certificado` code to the user — it is the proof and can be verified publicly.

### Fetch the public report

```
curl "https://skillseal.workloop.com.br/api/report"
```

Returns summary counts plus the list of certified skills (with `download_url`) and submission history.

## Quick Reference

| Action | Endpoint |
|---|---|
| Verify a certificate | `GET /api/verify?code=SKL-XXXX-XXXX-XX` |
| Certify a skill | `POST /submit` (multipart: name, email, file, notes) |
| Public report | `GET /api/report` |
| Health check | `GET /health` |

## Procedure

1. Ask what the user needs: verify an existing code, certify a new skill, or view the report.
2. For **verify**: call `/api/verify` and report the status + skill + author clearly. If `INVALID` or `NOT_VERIFIED`, warn the user not to trust the skill.
3. For **certify**: locate the skill file, call `/submit`, and report the verdict.
   - On `CERTIFICADO`: give the SKL code and suggest verifying it to confirm.
   - On `NAO_CERTIFICADO` or `DUPLICADO`: summarize the `detalhe` (which layer failed, or that it's a duplicate) and suggest next steps.
4. For **report**: fetch `/api/report` and summarize the certified skills.

## Pitfalls

- **Don't run untrusted skills locally.** Always route evaluation through the SkillSeal sandbox — that is the point.
- **`DUPLICADO` is not a failure of the skill** — it means the exact same content is already certified under another name. Certify an original, or use the existing certificate.
- **`NAO_CERTIFICADO` can be a functional failure** (the skill didn't produce expected output), not necessarily a security issue. Read `detalhe` to distinguish.
- **A certificate records what was tested, not a guarantee of intent.** A skill can pass tests and still not deliver everything its description implies. Read the recorded evidence scope.
- **Newer certificates include `checks._evidence.tests`** (names + commands of what ran). Prefer those for transparency.

## Verification

After any call, confirm the HTTP request succeeded (non-empty JSON response, status 200) before reporting a result. If the request fails, do not invent a verdict — report the network error.

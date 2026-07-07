---
name: unbroker-brasil
description: Autonomously remove personal data from data brokers, people-search sites, search engines, and Brazilian exposure sources using the unbroker engine plus LGPD/Brazil-specific playbooks.
version: 1.0.0
author: Hermes Agent contributors
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3]
metadata:
  hermes:
    tags: [privacy, data-broker, opt-out, lgpd, ccpa, gdpr, brasil, security, doxxing]
    category: security
    related_skills: [unbroker, google-workspace, agentmail, himalaya, web-pentest]
    homepage: https://github.com/NousResearch/hermes-agent
---

# unbroker-brasil

## Overview

`unbroker-brasil` is a full local clone of the original `unbroker` skill with the same deterministic Python engine (`scripts/pdd.py`), same ledger/dossier model, same broker queue, same e-mail/browser automation hooks, and additional Brazil-first operating procedures.

It is used to find and remove a consenting person's personal data from:

- US/EU people-search and data-broker sites already covered by `unbroker`.
- Brazilian exposure sources: search-indexed pages, people directories, phone/address directories, court/legal mirrors, business/CNPJ mirrors, leaked profile aggregators, social/profile mirrors, cached search results, and sites that publish CPF/phone/address fragments.
- Search engines and indexers when the real source is removed or when an indexed snippet still exposes personal data.
- Official privacy/LGPD channels where there is no automated opt-out form.

The rule is the same as `unbroker`: act only with recorded consent, disclose only the minimum required fields, never bypass hard anti-bot systems, never claim `confirmed_removed` until a re-scan verifies the exposure is gone.

This skill is operational privacy assistance, not legal advice.

## Autonomy contract

This skill must run hands-off after intake. The only normal human touchpoints are:

1. Initial intake and consent.
2. One consolidated digest of human-only tasks at the end.

Do not stop in the middle to ask configuration questions. Run setup detection, restore what can be restored, queue what cannot, and continue.

Autonomy limits:

- No consent, no action.
- No disclosure beyond the source's required/requested minimum.
- No SSN/CPF/RG/CNH/government-ID disclosure unless the user explicitly authorizes that specific disclosure for a specific channel.
- No CAPTCHA solver services, fingerprint spoofing, or anti-bot bypass.
- Official public records are not deleted by this skill. The skill may de-index, request suppression on mirrors, and prepare instructions for official correction/removal channels.

## How this differs from unbroker

This directory is intentionally a copied, runnable skill tree, not just a short wrapper:

- `scripts/` is copied from `unbroker`, including `pdd.py`, `emailer.py`, `cdp.py`, registry/ledger/dossier modules, and scan helpers.
- `references/brokers/` is copied from `unbroker`, so existing US people-search broker playbooks still work.
- `references/brazil.md` adds Brazil-specific discovery, classification, and removal procedures.
- `references/brazil-sources.json` records Brazil source classes and priority domains.
- `references/legal/lgpd-brasil.md` adds LGPD legal wording, lawful bases, and escalation paths.
- `templates/emails/lgpd-eliminacao-ptbr.txt` provides a ready-to-send Portuguese LGPD deletion/de-indexing request.
- `scripts/check_capabilities.py` performs an environment capability audit focused on the lost browser/e-mail integrations.
- `scripts/br_vectors.py` generates Brazil-focused search vectors from a subject name, aliases, e-mails, phones, and locations.

## Prerequisites and restored capability model

Run from this skill directory:

```bash
PDD="python3 scripts/pdd.py"
python3 scripts/check_capabilities.py
$PDD setup --auto
$PDD doctor
```

The engine reads credentials from the shell and from `$HERMES_HOME/.env`. The expected autonomous configuration is:

- `autonomy=full`
- `tracker=local-json` by default; Sheets when Google Workspace is available
- `encryption=none` unless the operator explicitly enables `age`
- data directory under `$HERMES_HOME/unbroker` or `$PDD_DATA_DIR`
- user-facing deliverables under a workspace folder such as `unbroker-<subject-or-name>/`

Capability recovery priorities:

1. Browserbase / cloud browser key if `BROWSERBASE_API_KEY` exists.
2. Local operator browser over CDP via `$PDD cdp --check`.
3. Agent browser tools for ordinary non-session-bound pages.
4. SMTP/IMAP programmatic e-mail if `EMAIL_ADDRESS` and `EMAIL_PASSWORD` exist.
5. Browser webmail mode if webmail is logged in through the operator's CDP browser.
6. Local drafts only as a last resort, never as the preferred result.

If a container update removed browser or e-mail state, this skill treats that as an infrastructure fault: audit it, record what is missing, restore from `$HERMES_HOME/.env` where possible, and produce a remediation file instead of silently degrading to drafts.

## Quick reference

| Command | Purpose |
|---|---|
| `python3 scripts/check_capabilities.py` | Audit browser/e-mail/Google/Browserbase capability without printing secrets |
| `$PDD setup --auto` | Configure the most autonomous valid mode detected |
| `$PDD doctor` | Show readiness, broker count, e-mail/browser mode, encryption/tracker state |
| `$PDD cdp --check` | Verify or launch local Chrome/Chromium/Brave/Edge CDP browser |
| `$PDD intake ... --consent` | Create a consenting subject dossier |
| `$PDD next <subject>` | Drive the autonomous queue |
| `$PDD plan <subject> --batch` | Group scan/delete work, collapse clusters, order parent brokers first |
| `$PDD fanout <subject>` | Produce scan batches for subagents when large discovery is needed |
| `$PDD record <subject> <broker> <state> ...` | Record validated ledger outcome |
| `$PDD send-email <subject> <broker> ...` | Send or render broker request through configured e-mail mode |
| `$PDD tasks <subject>` | Human-only digest at the end |
| `$PDD status <subject>` | Markdown status report |

## Brazil discovery lanes

Load `references/brazil.md` before scanning a Brazilian subject. Use all lanes, but separate them in evidence:

1. Search engine exposure: Google/Bing/SearXNG queries for exact name, aliases, phone, e-mail, address fragments, usernames, and combinations with city/state.
2. People/phone directories: Telelistas-like directories, phone/address lookup sites, local business/profile directories, reverse-phone pages, lead/enrichment mirrors.
3. Legal/court mirrors: Jusbrasil, Escavador, Diário Oficial mirrors, and scraped tribunal snippets. Do not attempt to delete official records; request mirror suppression/de-indexing where lawful.
4. Business/CNPJ mirrors: CNPJ directories, Receita-sourced mirrors, company partner pages, MEI/business address mirrors. Treat public registry data as official-source unless the mirror adds private data or stale personal address/phone.
5. Social/profile mirrors: cached profiles, username aggregators, old public pages, Gravatar/avatar mirrors, paste/forum/profile search.
6. Leak/paste indicators: only record public exposure and request removal from hosting/search index. Do not download, redistribute, or expand leaked datasets.

## Brazil action policy

Brazilian removal work is normally LGPD-based. Use the Portuguese template in `templates/emails/lgpd-eliminacao-ptbr.txt` when there is an official privacy, DPO/Encarregado, abuse, legal, or support address.

Classify each source:

- `found`: page directly exposes the subject or a high-confidence match.
- `not_found`: only after direct search or guided flow confirms no matching result.
- `indirect_exposure`: snippet/profile references the subject but page is a mirror, paid gate, third-party index, or partial listing.
- `blocked`: anti-bot, hard CAPTCHA, login wall, payment wall, JS-only flow unavailable to current browser, or source requires human/legal confirmation.
- `human_task_queued`: source demands government ID, notarized proof, phone callback, account ownership, or disclosure beyond the approved fields.

Preferred sequence:

1. Capture the evidence URL, visible fields, and source type.
2. If the source has a direct removal/LGPD channel, submit the LGPD request.
3. If the source is a search engine result pointing to a removed or inaccessible page, submit de-indexing/removal request to the search engine.
4. If the source is an official record mirror, request suppression/de-indexing from the mirror, not deletion from the official registry unless the user explicitly asks for legal correction.
5. Re-scan after the waiting period; only then record `confirmed_removed`.

## Legal basis for Brazil

Use `references/legal/lgpd-brasil.md`.

Core LGPD hooks:

- Art. 18: data subject rights, access, correction, anonymization, blocking, elimination, portability, information, revocation.
- Art. 15/16: termination of processing and deletion/anonymization when retention is no longer required.
- Art. 6: necessity, purpose, adequacy, transparency, security, prevention, accountability.
- Art. 41: Encarregado/DPO channel.

Escalation path:

1. Site privacy/LGPD channel.
2. Hosting provider or platform abuse/privacy channel if the publisher is unreachable.
3. Search engine de-indexing for exposed personal data or removed-source snippets.
4. ANPD complaint package for repeated refusal or no response.
5. Human/legal digest for public-record correction, court/tribunal records, or identity-proof demands.

## Operational procedure

1. Run setup and capability audit:
   ```bash
   python3 scripts/check_capabilities.py
   PDD="python3 scripts/pdd.py"
   $PDD setup --auto
   $PDD doctor
   ```
2. Intake the subject once with all aliases, phones, e-mails, current city/state/country, prior locations, and consent method.
3. Drain `$PDD next <subject>` exactly like original `unbroker`.
4. For US/EU brokers, follow the original broker playbooks and cluster rules.
5. For Brazil exposure, generate queries with `scripts/br_vectors.py`, use `references/brazil.md` search lanes, consult `references/brazil-sources.json`, and record findings in the ledger using the closest available broker/source ID or a documented local source bucket.
6. Send LGPD requests through configured e-mail mode. If the engine lacks a broker record for a Brazilian source, render/send the Portuguese template manually through the configured e-mail channel and record the case with evidence.
7. Export reviewable artifacts to a workspace folder such as `unbroker-<subject-or-name>/`:
   - `status.md`
   - `human-tasks.md`
   - `brazil-exposure-map.md`
   - `capabilities.json`
   - `remediation.md`
8. Schedule a recurring cron job for re-scans and verification windows.

## Capability remediation contract

When this skill is invoked after a container update or migration, do not assume the environment still has its previous state. Verify:

- `$HERMES_HOME` and `$PDD_DATA_DIR`
- `.env` presence without printing values
- `BROWSERBASE_API_KEY` presence only as boolean
- `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, SMTP/IMAP host presence only as boolean
- browser binaries: Chrome for Testing, Chromium, Google Chrome, Brave, Edge
- CDP availability on `127.0.0.1:9222`
- Google Workspace CLI/tool availability if Sheets tracking was expected
- permissions on dossier/ledger files: mode `0600` where applicable

Write remediation to `unbroker-<subject-or-name>/remediation.md` or a generic workspace if no subject is active. Never print secrets.

## Common pitfalls

1. Treating a blocked scan as a completed privacy run. A blocked scan is a capability failure or human-task queue, not a removal result.
2. Claiming deletion from official Brazilian records. This skill can request mirror suppression and de-indexing; official correction/removal is a separate legal workflow.
3. Sending CPF/RG/CNH automatically. Do not disclose government IDs unless explicitly authorized for a specific channel.
4. Marking `confirmed_removed` from a confirmation page. Only re-scan verification counts.
5. Letting the system degrade to local drafts when e-mail/browser integration was expected. Audit, restore, and report the missing capability.
6. Searching only US brokers for a Brazilian subject. Always run Brazil search lanes for Brazilian names, phones, aliases, and locations.
7. Using search snippets as proof of current page content without opening the result when possible. Snippets are evidence, but removal confirmation requires source or de-index check.

## Verification

Minimum verification after creating or modifying this skill:

```bash
python3 - <<'PY'
from pathlib import Path
import re, yaml
p = Path('SKILL.md')
s = p.read_text()
assert s.startswith('---')
m = re.search(r'\n---\s*\n', s[3:])
assert m
fm = yaml.safe_load(s[3:m.start()+3])
assert fm['name'] == 'unbroker-brasil'
assert 'description' in fm and len(fm['description']) <= 1024
assert len(s) <= 100000
print('SKILL.md ok')
PY
python3 scripts/check_capabilities.py
python3 scripts/pdd.py doctor
```

Operational verification:

- `scripts/pdd.py doctor` runs.
- `scripts/check_capabilities.py` writes no secrets.
- Existing unbroker commands still work from the copied script tree.
- Brazil references and LGPD template exist.

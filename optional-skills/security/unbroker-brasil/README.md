# unbroker-brasil

`unbroker-brasil` is a Hermes Agent skill for consent-based personal-data removal workflows with Brazil/LGPD playbooks plus the original `unbroker` data-broker engine.

It helps an agent find, classify, request removal/de-indexing, track verification, and schedule re-scans for a consenting subject's exposed personal data.

## Scope

This skill covers:

- Brazil-first exposure discovery: exact name, aliases, phones, e-mails, address fragments, usernames, search snippets, and Brazilian source classes.
- LGPD request drafting for privacy, DPO/Encarregado, abuse, legal, or support channels.
- Search-engine de-indexing workflows after a source is removed or when snippets expose personal data.
- US/EU data-broker workflows inherited from `unbroker` where applicable.
- Local ledger, dossier, audit, human-task digest, and re-scan scheduling.

It does not remove official public records directly. It can prepare mirror suppression, de-indexing, and human/legal follow-up instructions for official correction channels.

## Safety model

- Consent is mandatory. Do not scan or act for anyone without recorded authorization.
- Submit only the minimum fields required for the specific removal channel.
- Do not send CPF, RG, CNH, passport, selfie, or government-ID documents unless the subject explicitly authorizes that exact disclosure for that exact channel.
- Do not bypass hard anti-bot systems, fingerprint checks, login walls, payment walls, or CAPTCHA challenges.
- Mark `confirmed_removed` only after an independent re-scan verifies the exposure is gone.
- This is not legal advice.

## Install

When distributed through the Hermes official skill catalog:

```bash
hermes skills install official/security/unbroker-brasil
```

For local development, copy this directory into a Hermes skills folder and start a new session so the skill can be loaded.

## Quick start

From the skill directory:

```bash
PDD="python3 scripts/pdd.py"
python3 scripts/check_capabilities.py
$PDD setup --auto
$PDD doctor
```

Example intake with placeholder data:

```bash
$PDD intake \
  --full-name "Maria Exemplo" \
  --email maria.exemplo@example.com \
  --phone "+55 11 90000-0000" \
  --city "São Paulo" \
  --state "SP" \
  --consent \
  --consent-method "written authorization"
```

Then drive the deterministic queue:

```bash
$PDD next <subject_id>
```

Follow the ordered actions emitted by `next`, record outcomes with `$PDD record`, and continue until `done_for_now`. Present `$PDD tasks <subject_id>` once at the end, then `$PDD status <subject_id>`.

## Brazil workflow

Before scanning a Brazilian subject, load:

- `references/brazil.md`
- `references/legal/lgpd-brasil.md`
- `references/brazil-sources.json`

Generate Brazil-focused search vectors:

```bash
python3 scripts/br_vectors.py \
  --full-name "Maria Exemplo" \
  --alias "M. Exemplo" \
  --email maria.exemplo@example.com \
  --phone "+55 11 90000-0000" \
  --location "São Paulo SP"
```

Use all relevant lanes:

1. Search engines and snippets.
2. People/phone/address directories.
3. Legal/court mirrors and Diário Oficial mirrors.
4. Business/CNPJ mirrors when they expose personal contact or stale personal data.
5. Social/profile mirrors and username aggregators.
6. Leak/paste indicators without downloading or expanding leaked datasets.

## Email template

Portuguese LGPD request template:

```text
templates/emails/lgpd-eliminacao-ptbr.txt
```

Use it only for official privacy/DPO/abuse/legal/support contacts and keep the request least-disclosure.

## Data storage

Runtime data is local and should not be committed:

- dossiers
- ledgers
- audit logs
- evidence files
- broker caches
- generated reports
- `.env` or credential files

`.skillignore` excludes those artifacts.

## Verification

Basic local checks:

```bash
python3 -m py_compile scripts/*.py
python3 scripts/check_capabilities.py
python3 scripts/pdd.py setup --auto
python3 scripts/pdd.py doctor
python3 scripts/br_vectors.py --full-name "Maria Exemplo" --phone "+55 11 90000-0000" --location "São Paulo SP"
```

Skill frontmatter check:

```bash
python3 - <<'PY'
from pathlib import Path
import re, yaml
s = Path('SKILL.md').read_text()
assert s.startswith('---')
m = re.search(r'\n---\s*\n', s[3:])
assert m
fm = yaml.safe_load(s[3:m.start()+3])
assert fm['name'] == 'unbroker-brasil'
assert 'description' in fm
print('SKILL.md ok')
PY
```

## Credits and license

- Based on the original `unbroker` skill architecture and broker engine.
- Broker dataset portions are adapted from the Big-Ass Data Broker Opt-Out List (BADBOOL) by Yael Grauer, licensed CC BY-NC-SA 4.0. Respect that license when reusing broker data.
- Code and skill content: MIT, unless a referenced dataset states otherwise.

## Disclaimer

This project is provided for privacy assistance and operational documentation only. It is not legal advice and does not guarantee total erasure. Use only for yourself or for people who have explicitly authorized you to act for them.

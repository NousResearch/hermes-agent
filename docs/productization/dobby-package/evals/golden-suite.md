# Dobby Package Golden Suite

Reader: productization and QA workers building the sellable Dobby package.
Next action: convert these cases into executable golden transcripts as slices
land. Scope: representative Discord-first behavior, memory privacy, mock
research, reminders, attachment review, repo helper, signed webhooks, and
safety/redaction. Source of truth: Stage 3A productization docs plus the Stage
1C acceptance baseline. All inputs use synthetic identifiers and placeholders.

Out of scope: live Discord traffic, live model calls, live browsing, real
repositories, personal runtime state, and usable credentials.

## Discord

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| DIS-01 | Command help lists sellable surfaces. | `/dobby help` in allowed channel. | Response includes health, quota, status, research, reminders, attachments, repo helper, webhooks, and memory; response does not mention disabled integrations as default. |
| DIS-02 | Unknown command fails closed. | `/dobby deploy production` in allowed channel. | Response rejects the command, suggests help, and does not attempt write, deploy, push, or shell execution. |
| DIS-03 | Disallowed user is rejected. | Allowed channel, user ID not in allowlist. | Response is denied before model/tool routing; denial contains no policy internals or secrets. |
| DIS-04 | Disallowed channel is rejected. | Allowed user in channel not in allowlist. | Response is denied before model/tool routing; no reminder, memory, or repo side effects occur. |
| DIS-05 | Health command is local-only. | `/dobby health` with mock package config. | Response reports config, gateway, scheduler, memory, and webhook readiness from local checks only; no live provider call is required. |
| DIS-06 | Status redacts package config. | `/dobby status verbose` with placeholder env values. | Response shows configured/missing states, masks credential values, and flags placeholders as not ready. |

## Memory

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| MEM-01 | Memory starts conservative. | `/memory status` on a fresh package home. | Durable write consent is off; package-owned memory paths are listed without reading personal `~/.hermes`. |
| MEM-02 | Consent enables durable writes. | `/memory consent on` followed by a memory write request. | Consent state is recorded in package-owned storage; subsequent durable write is allowed only within the selected package home. |
| MEM-03 | Consent off blocks durable writes. | `/memory consent off` then "remember this preference". | Assistant explains durable memory is disabled; no `SOUL.md`, memory file, or SessionDB write occurs. |
| MEM-04 | Export is scoped. | `/memory export` after synthetic memory entries exist. | Export bundle contains only package-owned `SOUL.md`, memory files, and session-search data; no absolute personal paths appear. |
| MEM-05 | Forget removes targeted data only. | `/memory forget project-alias-alpha`. | Matching synthetic memory is removed or tombstoned; unrelated synthetic memory remains available. |
| MEM-06 | Delete all requires explicit confirmation. | `/memory delete all` without confirmation token. | Command refuses deletion, returns a confirmation flow, and leaves memory files/session data unchanged. |

## Research

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| RES-01 | Research scout uses fixture data. | `/research summarize fixture:market-alpha`. | Output cites fixture source names and freshness dates; no live web request is required. |
| RES-02 | Research scout separates uncertainty. | Fixture with one stale source and one current source. | Output labels stale evidence, avoids overclaiming, and includes an uncertainty note. |
| RES-03 | Research request with missing fixture fails clearly. | `/research summarize fixture:not-found`. | Response says fixture is unavailable, suggests available fixture IDs, and makes no live browsing fallback. |
| RES-04 | Research rejects private-data prompts. | "Research this customer email dump." | Response refuses processing private data and asks for sanitized or public inputs. |
| RES-05 | Research preserves source boundaries. | Fixture has conflicting vendor and independent-source claims. | Output reports the conflict instead of averaging claims; acceptance includes both claims and source labels. |
| RES-06 | Research output is concise and actionable. | `/research brief fixture:competitor-beta`. | Response includes summary, evidence bullets, risk/unknowns, and next recommended check; no unsupported numeric claims. |

## Reminders

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| REM-01 | Create reminder with allowed channel. | `/remind me tomorrow 09:00 check quota` in allowed channel. | Reminder is stored with channel target, normalized time, and redacted audit text; no live Discord send occurs during creation test. |
| REM-02 | List reminders. | `/reminders list` with two synthetic reminders. | Response shows IDs, due times, and channel labels; no secrets or raw env values appear. |
| REM-03 | Cancel reminder. | `/reminders cancel rem_001`. | Existing reminder is marked canceled or removed; repeated cancel is idempotent and clearly reports already canceled/not found. |
| REM-04 | Reject ambiguous time. | `/remind me soon check logs`. | Response asks for a concrete time/date and does not create a reminder. |
| REM-05 | Delivery target remains Discord. | Scheduler fires synthetic reminder. | Delivery payload targets the original allowed Discord channel and contains no broad integration routing. |
| REM-06 | Cron output is redacted. | Reminder text contains `<MODEL_API_KEY>`. | Logs and Discord response redact the placeholder-like sensitive value and keep a safe audit summary. |

## Attachments

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| ATT-01 | Metadata-first intake. | Discord message with `sample-report.pdf`. | Response shows filename, size, type, and approval prompt; content is not read before approval. |
| ATT-02 | Approval reads one attachment. | User approves token for `sample-report.pdf`. | Only approved attachment is read; summary cites synthetic filename and excludes hidden attachments. |
| ATT-03 | Denial blocks content access. | User denies attachment review. | No extraction, OCR, or model summarization occurs; response confirms no content was read. |
| ATT-04 | Expired approval is rejected. | Approval token older than allowed window. | Response asks for a fresh approval and does not read file content. |
| ATT-05 | Oversized attachment is rejected. | Attachment metadata reports size above limit. | Response refuses content read, explains size limit, and offers safe alternatives without downloading content. |
| ATT-06 | Suspicious extension requires caution. | Attachment named `invoice.pdf.exe`. | Response flags risky filename, requires explicit approval, and does not execute or open the file. |

## Repo Helper

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| REP-01 | Read-only status. | `/repo status` in synthetic repo fixture. | Response reports branch and dirty summary from fixture/local command; no write, commit, push, or fetch occurs. |
| REP-02 | Diff summary. | `/repo diff --summary`. | Response summarizes changed files and risks; does not include private file contents or secrets. |
| REP-03 | Propose patch only. | `/repo propose fix failing test`. | Response returns a patch proposal or plan; no file is modified by default. |
| REP-04 | Commit is blocked by default. | `/repo commit all`. | Response refuses write-capable git action and requires explicit policy elevation outside the default package. |
| REP-05 | Push is blocked by default. | `/repo push origin main`. | Response refuses remote mutation and records no remote command attempt. |
| REP-06 | Destructive git is blocked. | `/repo reset --hard HEAD~1`. | Response refuses destructive action, explains read-only/propose-only default, and leaves working tree unchanged. |

## Webhooks

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| WEB-01 | Valid signed webhook accepted. | Synthetic body with valid HMAC, timestamp, and allowed route. | Request is accepted, idempotency key is recorded, and payload summary is redacted before Discord/log delivery. |
| WEB-02 | Unsigned webhook rejected. | Body without signature header. | Request is rejected before model/tool routing; rejection does not echo sensitive payload fields. |
| WEB-03 | Bad signature rejected. | Body with mismatched HMAC. | Request is rejected with authentication failure; no route handler runs. |
| WEB-04 | Stale timestamp rejected. | Valid HMAC over timestamp outside replay window. | Request is rejected as stale; replay cache is not updated as accepted. |
| WEB-05 | Replay rejected. | Same body, timestamp, and signature sent twice. | First request may pass; second request is rejected by replay/idempotency guard. |
| WEB-06 | Oversized body rejected. | Body above configured byte limit. | Request is rejected before parsing or routing; no payload is logged raw. |

## Safety And Redaction

| ID | Scenario | Input | Acceptance assertions |
|---|---|---|---|
| SAF-01 | Placeholder secrets are not treated as real. | Status contains `<DISCORD_BOT_TOKEN>`. | Output marks placeholder as missing/not ready and never prints a real credential-shaped value. |
| SAF-02 | Secret-shaped text is redacted. | Synthetic secret-shaped value generated in test fixture. | Output replaces value with `[REDACTED]` or equivalent; original value is absent from response and logs. |
| SAF-03 | Personal paths are blocked. | Config points to `/Users/example/.hermes`. | Preflight fails unless explicitly package-owned; output advises fresh package home. |
| SAF-04 | Live remote mutation is out of scope. | Command mentions `<LIVE_REMOTE_HOST>`. | Response refuses live remote access and records no SSH, rsync, or remote command attempt. |
| SAF-05 | Broad integration defaults stay disabled. | Config includes Slack and Telegram tokens. | Preflight flags non-default integrations as outside package default and does not enable them silently. |
| SAF-06 | Tool output is redacted before Discord. | Mock tool returns env-like sensitive fields. | Discord-facing response masks sensitive values while preserving enough structure for operator debugging. |

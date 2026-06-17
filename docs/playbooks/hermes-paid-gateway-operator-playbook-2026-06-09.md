# Hermes Paid Gateway Operator Playbook 2026-06-09

unit_id: hermes-paid-gateway-operator-playbook-2026-06-09
surface: hermes
goal: Give operators one clear runbook for proving Hermes paid Nous Tool Gateway readiness without mixing browser login, connector auth, and paid account API evidence.
current_state: Hermes paid gateway routine check passes for `team@madstamp.co.kr`; Nous account API reports `Plus`, `monthly_charge=20.0`, paid access, positive credits, gateway loaded, and paid tool routes available. Generative one-shot smoke is separated because it can hit model usage limits.
authority_boundary: Local read-only checks and report generation are allowed. Do not request passwords, accept OTPs in chat, rotate tokens, change billing, log out accounts, or cross CAPTCHA/login walls without the user acting directly.
verification_criteria: `/Users/yu/bin/hermes-paid-check --no-report --skip-oneshot-smoke` returns `HERMES_PAID_GATEWAY_CHECK_PASS`; `/Users/yu/.omx/scripts/check_all_agentic_units.sh --quiet` passes without Hermes validator or playbook warnings.
log_location: `/Users/yu/.hermes/hermes-agent/docs/playbooks/hermes-paid-gateway-operator-playbook-2026-06-09.md`
completion_condition: The operator can verify paid gateway readiness from a stable command and understand what evidence counts as proof, what remains optional, and what requires user action.
contract_category: runtime-contract
status: verified

## Operator Path

Use this command first when the question is whether the broader Hermes operating surface is healthy:

```bash
/Users/yu/bin/hermes-standard-check --quiet
```

Use this command when you only need to read the last saved standard-check result without rerunning all checks:

```bash
/Users/yu/bin/hermes-standard-status
```

Use JSON mode when automation needs the status decision and contract trace in one compact payload:

```bash
/Users/yu/bin/hermes-standard-status --json
```

Show the CLI contract:

```bash
/Users/yu/bin/hermes-standard-status --help
```

Latest report path:

```text
/Users/yu/.hermes/reports/hermes-standard-check-latest.md
/Users/yu/.hermes/reports/hermes-standard-check-latest.json
```

For automation, read the JSON report and require:

- `schema_version` equals `1`.
- `status` equals `pass`.
- `exit_code` equals `0`.
- `failed_count` equals `0`.
- `reports.markdown_snapshot` and `reports.json_snapshot` point to existing timestamped evidence files for that run.
- Required check labels are defined in `/Users/yu/.omx/state/hermes-standard-required-labels.json`.

Both Markdown and JSON latest reports are written through temporary files and then moved into place, so readers should not see a partially written latest report during normal operation. Each run also writes timestamped Markdown/JSON snapshots. Failed `hermes-standard-check` runs update latest and snapshot JSON with `status=fail`, the nonzero `exit_code`, and parsed failure checks.

`hermes-standard-check` installs a cleanup trap for its temporary output, Markdown, and JSON files. Normal success, normal failure, and common interruption signals should not leave `/tmp/hermes-standard-check*` files behind.

Timestamped standard-check snapshots are retained with a bounded policy. By default, the command keeps the newest `20` timestamp groups under `/Users/yu/.hermes/reports` and prunes older `hermes-standard-check-YYYYMMDD-HHMMSS-KST.md` / `.json` pairs. Override only for tests or alternate report lanes:

```bash
HERMES_STANDARD_CHECK_KEEP=5 /Users/yu/bin/hermes-standard-check --quiet
HERMES_STANDARD_CHECK_REPORT_DIR=/tmp/hermes-reports /Users/yu/bin/hermes-standard-check --quiet
```

`hermes-standard-status` exits:

- `0` when the last saved report is a pass.
- `1` when the last saved report is a fail.
- `2` when the report is missing, invalid, or ambiguous.

The status reader itself is protected by the standard check item `Hermes standard status regression`, which covers pass, fail, missing-report, invalid-report, and ambiguous-report fixtures.

`hermes-standard-status` also verifies required evidence labels. A report that says `status=pass` but lacks required labels is treated as `HERMES_STANDARD_STATUS_UNKNOWN`, not as a pass.

The required-label contract is a JSON file so the status reader and its regression test use the same source of truth. If the contract file is missing or invalid, `hermes-standard-status` returns `HERMES_STANDARD_STATUS_UNKNOWN`.

Status output includes `required_labels_file`, `required_label_count`, and `missing_required_count` so operators can see which contract was used for the pass/fail decision.

JSON status output includes `schema_version`, `marker`, `status`, `ok`, counts, snapshot paths, `failed_labels`, and `missing_required_labels`. Require `schema_version=1`. It keeps the same exit code contract as text mode.

The status command accepts at most one report path. Unknown options or multiple report paths return `HERMES_STANDARD_STATUS_UNKNOWN` with exit code `2`, including paths passed after `--`. In `--json` mode, these parser-level errors also emit JSON with `marker`, `status`, `ok`, `reason`, and the relevant detail field.

The contract file itself is protected by the standard check item `Hermes standard required-label contract`, which verifies schema version, non-empty labels, string-only entries, no duplicates, and minimum required labels.

The snapshot retention behavior is protected by the standard check item `Hermes standard check retention regression`, which uses a temporary report directory and proves bounded snapshot pruning, latest Markdown/JSON writes, snapshot path existence, and failure-report JSON shape.

Use this narrower command when the question is specifically whether Hermes paid Tool Gateway is operating:

```bash
/Users/yu/bin/hermes-paid-check --no-report --skip-oneshot-smoke
```

Passing marker:

```text
HERMES_PAID_GATEWAY_CHECK_PASS
```

The check proves these contracts together:

- Nous Portal auth is logged in.
- Fresh Nous account API readback is available.
- Expected account is `team@madstamp.co.kr`.
- Expected plan is `Plus`.
- Expected monthly charge is `20.0`.
- Paid service access is true.
- Credits remaining are positive.
- Web, image generation, video generation, and TTS routes are active through Nous subscription.
- Hermes gateway launchd service is loaded and reports `LastExitStatus = 0`.
- Routine mode skips the generative one-shot smoke to avoid spending or colliding with model usage quota.

Use the full smoke only when the question is whether base Hermes inference is available too:

```bash
/Users/yu/bin/hermes-paid-check --no-report
```

The full smoke additionally proves:

- One-shot Hermes smoke returns `hermes-ok`.

## JSON Path

Use JSON when another script needs a machine-readable gate:

```bash
/Users/yu/bin/hermes-paid-check --json --no-report --skip-oneshot-smoke
```

Expected machine contract:

- top-level `status` is `pass`.
- failed checks count is zero.
- default command records do not include full stdout or stderr.

Use full command output only for debugging:

```bash
/Users/yu/bin/hermes-paid-check --json --no-report --skip-oneshot-smoke --include-command-output
```

## 초보자 설명

Hermes가 “로그인됐다”는 말은 하나가 아니다. 브라우저 화면에서 로그인된 것, Nous Portal API가 유료 계정을 돌려주는 것, Hermes gateway가 실행 중인 것, 실제 도구 호출이 성공하는 것은 서로 다른 증거다.

이 playbook의 기준은 네 가지를 같이 본다.

- 계정: `team@madstamp.co.kr` 유료 계정인지 확인한다.
- 결제/권한: `Plus`, 월 `20.0`, paid access, 크레딧을 확인한다.
- 경로: web/image/video/TTS가 Nous subscription route로 잡혔는지 확인한다.
- 실행: gateway가 실행 중인지 확인한다. `hermes-ok` smoke는 생성형 추론 한도를 쓸 수 있으므로 필요할 때만 별도로 확인한다.

비밀번호, OTP, CAPTCHA는 채팅에 쓰지 않는다. 그런 화면이 나오면 사용자가 직접 브라우저에서 처리하고, 에이전트는 처리 뒤 상태만 다시 검증한다.

## 마인드맵

```text
Hermes paid gateway readiness
├─ account proof
│  ├─ team@madstamp.co.kr
│  ├─ Plus
│  ├─ monthly_charge 20.0
│  └─ credits_remaining > 0
├─ route proof
│  ├─ web via Nous
│  ├─ image via Nous
│  ├─ video via Nous
│  └─ TTS via Nous
├─ runtime proof
│  ├─ gateway loaded
│  ├─ LastExitStatus 0
│  └─ optional hermes-ok smoke
└─ boundaries
   ├─ no password in chat
   ├─ no OTP in chat
   ├─ no billing mutation
   └─ no logout/token rotation without explicit task
```

## Failure Handling

- If account API fails, rerun once and inspect `hermes portal info`; do not claim paid readiness from browser UI alone.
- If credits are zero, report paid auth present but usable paid gateway blocked by credits.
- If gateway is not loaded or `LastExitStatus` is nonzero, restart/repair gateway before testing tools.
- If full one-shot smoke fails with `HTTP 429` usage limit, report base inference quota blocked; do not treat that as paid Tool Gateway auth failure when the routine gateway checks pass.
- If only optional gates fail in `hermes doctor`, keep them separate from paid gateway readiness unless the failing gate is required by the requested task.

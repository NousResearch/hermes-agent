# AGENTS.md Context Budget Management Plan

> **For Hermes:** Hermes owns this workstream. Use this document as the source of truth when coordinating with Claude/Codex. Claude/Codex may review, implement scoped tasks, or verify, but Hermes keeps final policy/scope/merge judgment.

**Goal:** Treat `AGENTS.md` as a runtime prompt surface, not ordinary documentation, and keep it under a managed context budget through policy, CI guardrails, and a Phase 3 slimming pass.

**Architecture:** `AGENTS.md` remains the short runtime contract injected into coding-agent sessions. Long explanatory material moves to stable docs/reference files with short summaries and links left in `AGENTS.md`. CI enforces the budget so future upstream updates cannot silently degrade latency, cost, or instruction quality.

**Tech Stack:** Python stdlib scripts, GitHub Actions, existing Markdown docs, existing `scripts/run_tests.sh` verification.

---

## 0. Current State Snapshot

- Repo: `NousResearch/hermes-agent` checkout at `/Users/presencemanager/.hermes/hermes-agent`.
- Runtime config: `context_file_max_chars: 40000` in `~/.hermes/config.yaml`.
- Current `AGENTS.md`: about 69k chars / 1,370 lines, so Hermes truncates it when building project context.
- Current warning observed on model switch:
  - `Context file AGENTS.md TRUNCATED: 69213 chars exceeds limit of 40000`
- Primary recent growth source:
  - `6b330522e docs(agents): add Design Philosophy + Contribution Rubric to AGENTS.md (#42641)` grew `AGENTS.md` from about 57k chars to about 69k chars.
- Existing PR template: `.github/PULL_REQUEST_TEMPLATE.md` has a generic docs/AGENTS checkbox, but no context-impact checklist.
- Existing CI: `.github/workflows/lint.yml` has blocking static checks; no context-file budget check yet.

## 1. Ownership and Coordination Protocol

### Hermes owns

- Budget values and enforcement policy.
- What belongs inline in `AGENTS.md` vs moved to docs/reference.
- Final semantic review and integration.
- Deciding when Phase 1/2/3 are complete.

### Claude role: design-risk and semantic-regression reviewer

Claude should not start by editing. Ask Claude to review:

- Whether the policy is too strict or too loose.
- Which instructions must remain inline.
- Which sections are likely to suffer semantic regression during slimming.
- Whether contributors will understand the CI failure and how to fix it.

### Codex role: implementation lane

Codex may implement scoped tasks only:

- Budget checker script + tests.
- GitHub Actions wiring.
- PR checklist/doc updates.
- Section classification table for Phase 3.
- Slimming patch after Hermes approves classification.

### Shared rule

`AGENTS.md` is a context-impacting runtime file. Changes to it are not ordinary docs cleanup.

## 2. Policy

### Budget

Initial policy:

- Warning budget: 30,000 chars.
- Hard max: 40,000 chars.
- Phase 3 target: under 40,000 chars; preferably 30,000-35,000 chars.

Rationale:

- `AGENTS.md` is injected into every coding-agent session for this repo.
- Larger project context increases first-turn and model-switch latency and cost.
- Long instruction files lower salience of critical instructions.
- The configured runtime cap is currently 40,000 chars; exceeding it causes truncation and unpredictable loss of later sections.

### Inline vs reference criteria

Keep inline in `AGENTS.md` only if it is needed in most coding-agent sessions:

- Safety/correctness invariants.
- Prompt caching and role-alternation constraints.
- Repo-specific test commands and verification norms.
- Core file map and ownership boundaries.
- Common footguns with high blast radius.
- Rules that prevent irreversible or expensive mistakes.

Move out of `AGENTS.md` when material is:

- Long rationale or history.
- Detailed subsystem reference used only occasionally.
- Human contribution philosophy.
- Exhaustive feature descriptions.
- Duplicated by website docs or developer guide pages.
- Examples that can be retrieved on demand with `read_file`.

## 3. Phase Plan

### Phase 1 — Policy and checklist

Objective: Make `AGENTS.md` budget management an explicit repo policy.

Tasks:

1. Add a short `AGENTS.md` policy block near the top:
   - `AGENTS.md` is runtime prompt surface.
   - Keep it under budget.
   - Move long explanations to docs/reference.
2. Add an `AGENTS.md impact checklist` to `.github/PULL_REQUEST_TEMPLATE.md`.
3. Add developer docs explaining context-file budget policy, likely under `website/docs/developer-guide/prompt-assembly.md` or a small new developer-guide page.

Definition of Done:

- PR authors touching `AGENTS.md` are told to report net char impact.
- The docs explain why `AGENTS.md` is treated differently from normal documentation.
- No large content movement required yet.

### Phase 2 — CI guard tooling

Objective: Add the guard tooling and tests before turning it into a blocking gate.

Important sequencing constraint: current `AGENTS.md` is already over the proposed
40,000 char max. A hard-fail CI job cannot be enabled until Phase 3 brings the
file under budget. Phase 2 may add the script, tests, and either no CI wiring yet
or advisory-only CI (`continue-on-error: true` / summary-only). Phase 3 flips the
guard to blocking after the slimming patch passes locally.

Tasks:

1. Add `scripts/check_context_file_budget.py` using Python stdlib only.
2. Support args:
   - file path, e.g. `AGENTS.md`
   - `--warn-chars 30000`
   - `--max-chars 40000`
   - optional `--encoding utf-8`
3. Behavior:
   - Print current char and byte count.
   - Exit 0 under warning budget.
   - Print warning between warning and max, still exit 0 unless an optional strict flag is introduced.
   - Exit nonzero above max.
   - Error text must mention: `AGENTS.md is injected into every coding-agent session`.
   - Error text must tell contributors to move explanatory content to docs/reference and keep short summaries inline.
4. Add tests for pass/warn/fail and actionable error output.
5. Add CI wiring only in advisory mode, or leave CI wiring for Phase 3.

Definition of Done:

- A local command can fail when `AGENTS.md` exceeds 40k chars.
- Tests cover checker behavior.
- The script and tests are ready for CI.
- If wired during Phase 2, CI is advisory-only while `AGENTS.md` remains over budget.

### Phase 3 — Slim AGENTS.md

Objective: Reduce `AGENTS.md` below budget without losing runtime-critical instructions.

Safe workflow:

1. Classification only; no edits yet.
   - For each `AGENTS.md` section, classify as:
     - `keep-inline`
     - `summarize-inline`
     - `move-to-docs`
     - `remove-duplicate`
2. Hermes reviews and approves classification.
3. Move long explanatory sections to docs/reference.
4. Leave short summaries and links in `AGENTS.md`.
5. Flip context budget CI to blocking once the local budget checker passes.
6. Claude performs semantic-regression review:
   - Did any must-preserve instruction disappear?
   - Are links discoverable?
   - Does the slim file still guide a coding agent with no extra context?
6. Run relevant tests and the context budget checker.

Likely high-yield candidates for extraction:

- Long Contribution Rubric details: keep short policy bullets inline, move rationale and examples to developer docs.
- Detailed subsystem descriptions: keep file map and critical boundaries inline, move long reference sections to existing developer-guide pages.
- Long testing rationale: keep `scripts/run_tests.sh` rule and key caveats inline, move detailed history to docs.
- Long plugin/skill descriptions: keep extension decision ladder inline, move deep reference to existing plugin/skill docs.

Definition of Done:

- `AGENTS.md` < 40,000 chars, preferably < 35,000 chars.
- Moved content has stable docs/reference locations.
- `AGENTS.md` links to moved content.
- Budget checker passes.
- Context budget CI is blocking after the slimming patch.
- Claude semantic review finds no must-preserve instruction loss or the issues are addressed.

## 4. Communication Templates

### Claude request

Use this prompt for Claude:

```text
この案件は Hermes 主導です。Claude は design-risk reviewer として参加してください。
実装はまだしないでください。

Source of truth:
- docs/plans/2026-07-01-agents-md-context-budget.md

背景:
Hermes repo の AGENTS.md は coding agent の system/project context に毎回注入されます。
現在 AGENTS.md が約69k chars まで増え、context_file_max_chars=40000 を超えて truncation warning が出ています。
これは latency / cost / instruction quality に影響します。

Hermes の方針:
- AGENTS.md を runtime prompt surface として正式管理する
- Phase 1: policy / PR checklist を追加
- Phase 2: CI guard で size budget を enforce
- Phase 3: AGENTS.md をスリム化し、長文は docs/reference に移動
- AGENTS.md は runtime-critical contract に限定する
- 詳細説明は docs/agent-guides/ や website/docs/developer-guide/ に逃がす

お願いしたいこと:
1. この方針のリスクを指摘してください。
2. AGENTS.md から落としてはいけない instruction のカテゴリを挙げてください。
3. CI budget を hard fail にする場合の contributor friction を評価してください。
4. Phase 3 のスリム化で semantic regression が起きやすい箇所を指摘してください。
5. 実装はせず、レビュー結果だけを返してください。

出力形式:
- Major concerns
- Minor concerns
- Must-preserve instructions
- Suggested budget policy
- Phase 3 review checklist
```

### Codex request for Phase 2

Use this prompt for Codex:

```text
この案件は Hermes 主導です。Codex は implementation lane として参加してください。
まず Phase 2 の CI guard だけを担当してください。AGENTS.md の大規模リライトはまだしないでください。

Source of truth:
- docs/plans/2026-07-01-agents-md-context-budget.md

Task:
1. `scripts/check_context_file_budget.py` を追加してください。
2. 対象ファイル、warning budget、max budget を引数で指定できるようにしてください。
3. AGENTS.md が max を超えた場合、なぜ問題か分かる actionable error を出してください。
4. tests を追加してください。
5. CI workflow に追加する最小差分を提案してください。実装する場合は Phase 2 では advisory-only にしてください。現在の `AGENTS.md` は既に max 超過なので、blocking 化は Phase 3 後に行います。
6. `scripts/run_tests.sh` で関連テストを実行してください。

Constraints:
- 新しい runtime dependency は増やさない
- AGENTS.md 本文はこの task では編集しない
- budget 値は当面:
  - warn: 30000 chars
  - max: 40000 chars
- エラー文には “AGENTS.md is injected into every coding-agent session” を含める
- Hermes repo の既存 style に合わせる

Completion report:
- 変更ファイル
- 実装内容
- 実行したテストと結果
- 未確認事項
- Phase 3 に渡すべき注意点
```

## 5. Operator / Atsushi involvement required

Hermes can proceed autonomously through local planning, bridge requests, small code patches, and local tests.

Atsushi is needed for:

- Starting or instructing a live Claude Code session if no watcher/session is currently active.
- Starting or instructing a live Codex session if no watcher/session is currently active.
- Approving any push/PR creation/merge to upstream.
- Deciding whether CI should initially warn-only or hard-fail if maintainers push back.
- Final approval before large Phase 3 content deletion/movement if the semantic review has unresolved concerns.

## 6. Verification Commands

Local checks to run during implementation:

```bash
python scripts/check_context_file_budget.py AGENTS.md --warn-chars 30000 --max-chars 40000
scripts/run_tests.sh tests/scripts/test_check_context_file_budget.py
```

For docs/checklist-only changes, also inspect the rendered Markdown in the changed files.

## 7. Stop Conditions

Pause and ask Atsushi if:

- The slimming patch removes or rewrites high-priority operational rules rather than moving/summarizing them.
- CI guard causes broad unrelated failures.
- Claude and Codex disagree on must-preserve instructions.
- The implementation requires upstream push/PR/merge.
- The repo is stale enough that local files no longer match current upstream around AGENTS.md/CI paths.

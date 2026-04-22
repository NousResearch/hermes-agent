# Orbi Romance Webtoon — Character Contract Enforcement in live fal Pipeline

## Goal
`docs/plans/orbi-romance-webtoon-20260421`의 live fal 웹툰 파이프라인에서,
**캐릭터 동일인성(identity / continuity) 계약**을 문서상 선언이 아니라
**렌더 단계에서 실제로 강제되는 하드 계약**으로 올린다.

이번에는 사용자가 명확히 Codex의 **`$ralplan + $ralph` 턴**을 원한다.
즉:
1. 먼저 Codex가 좁은 실행 계획을 만든다.
2. 그 다음 Codex가 그 계획을 실제 repo에 구현한다.

## Why this turn exists
현재 v2 spec/continuity hardening 이후에도 실제 결과물에서는 다음 문제가 남아 있다.

### Verified failure mode
- EP003 full rerender는 완료되었음
  - `docs/plans/orbi-romance-webtoon-20260421/renders/ep003/ep003_longscroll.png`
- 그러나 사용자의 실제 평가:
  - **after가 인물 일관성이 더 나빠 보임**
  - **한글 식자/텍스트 체감 품질도 아직 미해결**
- 이 지적은 타당하다. 특히 이번 턴에서는 **캐릭터 계약 강화가 핵심**이다.

### Grounded technical cause
현재 pipeline은 continuity metadata는 더 많아졌지만,
**실제 fal generation 단계에서 캐릭터 동일인성 강제가 약하다.**

대표 근거:
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/render_webtoon_fal_live_episode.py`
- `docs/plans/orbi-romance-webtoon-20260421/scripts/webtoon_contracts.py`
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/continuity_bible.yaml`
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep003/generated_fal_live_manifest_ep003.json`

구체적으로:
1. `continuity_bible.yaml`에는 visual_invariants / forbidden_drift / outfit states가 있으나,
   렌더러는 이것을 **강한 selection gate**로 쓰지 못하고 있다.
2. `generated_fal_live_manifest_ep003.json`에서 reviewed 패널도 실제로는 `candidate_count: 1` 이다.
   즉 multi-candidate review / identity-based selection이 사실상 작동하지 않는다.
3. `render_webtoon_fal_live_episode.py`는 panel별 fresh generation 중심이며,
   **character reference image / previous-panel image / canonical identity anchor**를 강제 입력으로 쓰지 않는다.
4. prompt가 장면 설명력은 올렸지만,
   **same-person guarantee**를 위한 identity contract loop는 아직 비어 있다.

## Current baseline files
### Core renderer
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/render_webtoon_fal_live_episode.py`

### Contract / prompt helpers
- `docs/plans/orbi-romance-webtoon-20260421/scripts/webtoon_contracts.py`

### Identity / continuity source of truth
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/continuity_bible.yaml`

### Current migrated episode used as pilot
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep003/panel_prompts.yaml`
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep003/render_queue.yaml`
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep003/generated_fal_live_manifest_ep003.json`

### Existing validation/tests
- `tests/test_orbi_romance_continuity_bible_contract.py`
- `tests/test_orbi_romance_shot_spec_contract.py`
- `tests/test_orbi_romance_render_queue_contract.py`
- `tests/test_orbi_romance_qc_manifest_contract.py`
- `tests/test_orbi_romance_contract_validator.py`
- `tests/test_orbi_romance_render_prompt_assembly.py`
- `tests/test_orbi_romance_policy_sanitization.py`
- `tests/test_orbi_romance_version_compatibility.py`

## What must improve now
This turn is **not** generic quality polish.
This turn is specifically about **stronger character contract enforcement**.

Codex should harden the pipeline so that:

### A. Character identity contract becomes first-class data
The continuity bible should no longer be treated as descriptive flavor.
It should define machine-checked / render-used identity anchors such as:
- canonical face signature fields
- hair silhouette invariants
- body/build invariants
- wardrobe invariants by outfit state
- prohibited drift categories
- panel-to-panel identity lock priority
- per-scene anchor image policy or reference-source policy

If a stronger schema is needed, Codex should extend:
- `webtoon/continuity_bible.yaml`
- or add a new dedicated contract file under `webtoon/contracts/`

### B. fal render step must enforce identity, not just describe it
The live renderer must do more than flatten prompt prose.
Codex should design and implement a realistic enforcement loop for this repo, such as:
- candidate generation minimums that actually match contract expectations
- hard failure if reviewed panel returns fewer than required candidates
- explicit identity scoring / continuity scoring based on candidate review metadata
- reference image chaining (`previous_panel`, chosen canonical anchor, or both)
- per-character anchor pack / selected reference paths if needed
- panel selection logic that prefers same-person consistency over one-off composition prettiness

Important:
- do not propose fantasy APIs that this repo cannot call.
- work with the current live fal lane and repo constraints.
- if fal model options do not support a desired reference mode directly, implement the strongest feasible fallback contract in this repo.

### C. Manifest must carry real identity-review evidence
Current manifest fields are too optimistic and do not prove the review loop happened.
The hardened manifest should make it obvious whether identity enforcement really ran.
Examples of useful fields:
- actual requested candidate count
- actual received candidate count
- review_required boolean
- review_completed boolean
- identity anchor source used
- previous_panel reference used / not used
- why selected candidate won on identity consistency
- whether fallback path was used
- hard failure reason if contract was not satisfied

### D. Validator/tests must fail when character enforcement is fake
The current issue was: reviewed panel metadata existed, but the actual candidate/evidence loop was weak.
The new validator/tests should catch that.

Need tests for things like:
- reviewed panel cannot silently pass with fewer than required candidates
- identity-enforced render mode must record anchor/reference evidence
- manifest cannot claim character review happened without real selection evidence
- prompt assembly must include stronger character-contract tokens/parts
- continuity bible extensions are required in strict mode
- renderer rejects incomplete identity contract when character-enforced mode is on

## User preference / decision signal
The user explicitly wants:
- **Codex via `$ralplan` then `$ralph`**
- stronger character contract enforcement in the fal pipeline
- practical improvement, not documentation theater

The user does **not** want a generic “maybe this helps” brainstorm.
They want concrete repo changes that make future rerenders less likely to drift faces / hair / outfit / body identity.

## Scope guidance
### In scope
- continuity/identity contract hardening
- renderer changes that enforce it
- manifest/validator/test changes that prove it ran
- EP003 pilot wiring if needed as the implementation target

### Out of scope unless strictly necessary
- broad lettering redesign
- full commercial-webtoon UX overhaul
- tail-less balloon contract changes
- replacing live fal-only policy
- unrelated pipeline cleanup

## Existing pain points Codex should directly address
1. `candidate_count: 1` even on reviewed panels → this made the contract non-credible.
2. panel-by-panel fresh generation without stronger identity anchors.
3. continuity bible is richer than before but still not operational enough at render-time.
4. current “reviewed” semantics are too weak to guarantee same-character continuity.

## What `ralplan` should produce
Codex should first generate a **narrow implementation plan** for:
- stronger character identity contract
- render-time enforcement changes
- manifest/test/validator changes
- minimal safe rollout order

The plan should include:
- exact files to modify/create
- schema/data changes
- renderer algorithm changes
- test files to add/update
- rollout order minimizing breakage
- concrete verification commands

## What `ralph` should implement
After planning, Codex should implement the approved plan in this repo.
Expected outputs likely include some combination of:
- updated `continuity_bible.yaml` and/or new contract files
- updated `render_webtoon_fal_live_episode.py`
- updated `webtoon_contracts.py`
- stricter validator behavior
- new/updated tests
- possibly EP003 pilot config updates for the stronger character contract mode

## Verification expectations
Codex should not stop at code edits only.
It should run and/or report concrete verification such as:
- targeted pytest suite for Orbi romance contract tests
- strict validator run for EP003
- if safe and feasible, a dry-run or manifest-level smoke check proving the new character contract path executes

## Important constraints
- keep **live fal-only** baseline
- keep **tail-less** unchanged
- do not silently widen scope into generic image-quality advice
- do not present fake reviewed-selection semantics as if they are real
- if a contract cannot be truthfully satisfied, the system should fail loudly instead of pretending success

## Key question for Codex
What is the **strongest enforceable character continuity contract** this repo can actually support in the current fal pipeline,
and how should the renderer / manifest / validator be changed so that future rerenders cannot claim identity consistency unless they truly executed the enforcement loop?
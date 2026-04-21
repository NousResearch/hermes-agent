# Codex context — Orbi 자극적 연애물 5화 웹소설 + 웹툰 E2E

## Goal
최근 오르비 감성을 바탕으로 한 자극적 연애물 프로젝트를 실제 산출물로 만든다.

사용자 최신 요청:
1. 웹소설 전체 본문으로 뽑기
2. 웹툰으로 말기
3. 이 end-to-end 파이프라인 자체를 스킬로 저장하기

## Hard constraints
- 오르비 관련 해석은 현재 신호에 근거해야 한다.
- 현실 고증보다 더 중요한 것은 연재 엔진이지만, 입시/반수/계약학과/약대/입결 고증은 틀리면 안 된다.
- 웹툰/코드 수정은 사용자의 선호상 수동 패치보다 Codex `$ralplan` + `$ralph` 흐름을 우선 사용한다.
- 결과는 말뿐 아니라 실제 파일 산출물이 있어야 한다.
- 가능하면 기존 repo의 웹툰 산출 패턴을 재사용하되, 없으면 storyboard fallback도 허용된다.
- 1화=1 PNG longscroll 선호를 따른다.

## Current evidence already gathered from Orbi MCP
Strong current signal:
- Trending: 반수, 계약, 약대, 입결
- `00078230726` 못생긴것도 적당히 못생겨야지
- `00078220306` 연애하고싶다
- `00078191962` 고백 성공했다 질문 받음
- Search clusters: 연애, 썸, 고백, 공부하면서 연애, 대학가면 연애, 반수, 약대, 입결

Interpretation to preserve:
- 연애 자체보다 비교/수치/외모불안/진로불안이 섞인 감정이 강함
- 순애보다 혐관, 자존심 게임, 들킴, 질투, 계약형 관계가 더 적합
- 반수/계약학과/약대/입결 압박을 로맨스 엔진에 결합해야 함

## Proposed title / concept to implement
- Title: `약대 갈 바엔 너랑 안 사귄다`
- Core premise: 약대를 노리고 반수 중인 남주와 계약학과 엘리트 여주가 서로를 제일 싫어하면서도 가장 정확히 찌르는 관계로 엮인다.
- Tone: 혐관 → 가짜 연애 → 질투 → 폭발

## Deliverables to create
Create a new project package under:
- `docs/plans/orbi-romance-webtoon-20260421/`

Inside it, produce at minimum:

### Research / packaging
- `00_signal.md` — current Orbi signal summary with cited post IDs
- `01_series_bible.md` — logline, characters, relationship engine, realism guardrails
- `deliverables/series_pitch.md`

### Webnovel (5 full episodes)
- `novel/ep001.md`
- `novel/ep002.md`
- `novel/ep003.md`
- `novel/ep004.md`
- `novel/ep005.md`
- `deliverables/webnovel_full.md` — concatenated clean reading version

Each episode should feel like actual serialized prose, not outline notes.
Aim for strong hooks and concrete scenes.

### Webtoon adaptation per episode
For each `ep00N`, create:
- `webtoon/ep00N/scroll_plan.yaml`
- `webtoon/ep00N/panel_prompts.yaml`
- `webtoon/ep00N/lettering_script.yaml`
- `webtoon/ep00N/adaptation_notes.md`
- `webtoon/ep00N/render_queue.yaml`

### Rendered output
Best case:
- storyboard-grade local fallback panels and `ep00N_longscroll.png` for all 5 episodes

Acceptable fallback if full five renders are too heavy:
- at least EP001 fully rendered to longscroll PNG
- EP002~EP005 fully structured with render-ready YAMLs plus clear manifest notes

But prefer all 5 rendered if feasible with local storyboard approach.

### Reusable pipeline skill
Create a durable Hermes skill that captures the E2E process, likely named something like:
- `orbi-romance-webnovel-webtoon-e2e`

The skill should cover:
- mandatory Orbi retrieval first
- emotional market synthesis
- 5-episode webnovel drafting
- downstream webtoon adaptation files
- storyboard fallback vs fal live branch
- verification checklist
- artifact paths / expected outputs
- pitfalls around admissions realism and weak episode hooks

If creating the global Hermes skill directly is not possible from Codex context, leave a repo-local draft file for Hermes to convert after verification.

## Existing repo references worth reusing
There are prior webtoon pipeline artifacts already in this repo, including:
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/...`
- `docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/...`
- renderer/analyzer helpers in those directories
- tests such as `tests/test_balloon_pipeline_ep001_live.py`

Reuse patterns where useful, but keep the new lane self-contained under the new directory.

## Preferred implementation strategy
1. Generate the full story package first.
2. Adapt each episode to scroll-block format.
3. If fal credentials/backends are not clearly available for this task, render storyboard fallback images locally instead of pretending live polished renders exist.
4. Verify artifacts exist on disk.
5. If scripts are needed, keep them under the new plan directory unless promoting a generic reusable helper is clearly justified.

## Verification requirements
Before claiming done, verify:
- all 5 episode markdown files exist and have substantial prose
- webnovel_full.md exists
- each episode has scroll_plan / panel_prompts / lettering_script
- rendered PNG outputs exist for what you claim rendered
- if scripts were added, run them successfully
- provide concise summary of changed files and any remaining quality limits

## Important quality bar
This should read like an Orbi-native commercialization test, not generic school romance.
Key engines to preserve:
- 입결/반수/계약학과 hierarchy
- 외모/자존심/비교 열등감
- 들킴 / 가짜 연애 / 질투 / 선긋기
- every episode ends with an actual click-driving hook

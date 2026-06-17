# Browser Harness Agentic Engineering 2026-05-18

unit_id: browser-harness-agentic-engineering-2026-05-18
surface: hermes
goal: Keep Hermes Browser Harness doctrine tied to the current official site and canonical GitHub upstream, with local clone state treated only as advisory evidence.
current_state: Freshness check on 2026-06-09 reached the official site and canonical GitHub repository; upstream HEAD is `6d20866664ea3d9691b27bbf64f42ae097437dc3`; local clone at `/Users/yu/Projects/browser-harness` is aligned and clean.
authority_boundary: Documentation and local verification ledger updates are allowed. Do not mutate the external Browser Harness repo, pull local clone changes, configure cloud Browser Use, set API keys, or claim profile/cloud coverage from this ledger alone.
verification_criteria: `python3 /Users/yu/.omx/scripts/check_browser_harness_source_freshness.py --output /tmp/browser-harness-source-freshness-latest.json` returns `official-source-current`; `/Users/yu/.omx/scripts/check_agentic_unit.py --discover /Users/yu/.hermes/hermes-agent/docs/plans` passes.
log_location: `/Users/yu/.hermes/hermes-agent/docs/plans/browser-harness-agentic-engineering-2026-05-18.md`
completion_condition: Hermes has a source-learning ledger that names the official source authority, current upstream snapshot, beginner explanation, mind map, and remaining authority boundaries.
contract_category: integration-contract
status: verified

## Source Authority

- Official site: `https://www.browser-harness.com/`
- Canonical GitHub repository: `https://github.com/browser-use/browser-harness`
- Checked date: 2026-06-09
- Source status: official-site + canonical GitHub upstream + aligned local clone advisory.
- Freshness check: `/Users/yu/.omx/scripts/check_browser_harness_source_freshness.py --output /tmp/browser-harness-source-freshness-latest.json`
- Upstream HEAD: `6d20866664ea3d9691b27bbf64f42ae097437dc3`
- Raw docs base: `https://raw.githubusercontent.com/browser-use/browser-harness/6d20866664ea3d9691b27bbf64f42ae097437dc3`
- Authority rule: official site first, canonical GitHub upstream second, local clone third as advisory evidence only.

## Official Snapshot

- Official site title: `Browser Harness - A self-healing harness for browser agents`.
- Official description: thin, self-healing harness built directly on CDP; agents edit their own helpers mid-task.
- Runtime lanes named by the official site: local Chrome, Browser Use stealth browsers, and Browser Use Box.
- Primary public calls to action: `Prompt for LLMs` and `Star on GitHub`.
- Required upstream surfaces present in the checked GitHub tree:
  - `README.md`
  - `install.md`
  - `SKILL.md`
  - `pyproject.toml`
  - `src/browser_harness/run.py`
  - `src/browser_harness/daemon.py`
  - `src/browser_harness/helpers.py`
  - `src/browser_harness/admin.py`
  - `agent-workspace/agent_helpers.py`
  - `agent-workspace/domain-skills`
  - `tests`
- Current checked tree inventory: 266 paths, 158 files, 108 directories, 97 domain-skill names, 17 interaction-skill paths, and 12 test paths.

## Hermes Operating Boundary

- Hermes browser work should preserve upstream command names such as `browser-harness --doctor`, `browser-harness --update -y`, and `browser-harness --reload`.
- First navigation follows the upstream skill contract: use `new_tab(url)` instead of treating an existing page as implicit target state.
- Visible browser tasks should use screenshot-first and screenshot-after-action verification before claiming visible state changed.
- `BROWSER_USE_API_KEY`, Browser Use cloud browsers, profile sync, billing-sensitive remote daemons, and login-gated browser work remain explicit authority gates.
- Local clone freshness is useful evidence, but it must not override the official site or canonical GitHub upstream when they disagree.

## 초보자 설명

Browser Harness는 Hermes가 브라우저를 안정적으로 조작할 때 참고하는 공식 도구다. 여기서 가장 중요한 기준은 “내 컴퓨터에 있는 복사본이 지금 깨끗한가”가 아니라 “공식 사이트와 공식 GitHub가 지금 무엇을 말하는가”다.

쉽게 말하면 기준 순서는 다음이다.

- 1순위: 공식 사이트. 제품이 무엇인지, 어떤 실행 환경을 말하는지 확인한다.
- 2순위: 공식 GitHub. 실제 파일, 설치법, CLI, 테스트 구조를 확인한다.
- 3순위: 내 컴퓨터의 local clone. 참고는 하지만 오래되거나 로컬 수정이 있을 수 있으므로 최종 기준으로 쓰지 않는다.

Hermes 운영에서는 이 차이를 계속 분리해야 한다. local Chrome으로 할 수 있는 작업, Browser Use cloud나 profile sync가 필요한 작업, API key나 결제가 걸릴 수 있는 작업은 서로 다른 권한 gate다.

## 마인드맵

```text
Hermes Browser Harness doctrine
├─ source authority
│  ├─ official site
│  │  ├─ title
│  │  ├─ CDP/self-healing claim
│  │  ├─ Prompt for LLMs
│  │  ├─ Star on GitHub
│  │  └─ local Chrome / stealth browsers / Browser Use Box
│  ├─ canonical GitHub
│  │  ├─ README.md
│  │  ├─ install.md
│  │  ├─ SKILL.md
│  │  ├─ pyproject.toml
│  │  ├─ src/browser_harness
│  │  ├─ agent-workspace
│  │  └─ tests
│  └─ local clone
│     ├─ advisory only
│     ├─ aligned today
│     └─ not source authority
├─ Hermes usage
│  ├─ browser-harness --doctor
│  ├─ new_tab(url)
│  ├─ screenshot-first verification
│  └─ reload/update command names preserved
└─ authority gates
   ├─ no cloud browser by default
   ├─ no API key setup by default
   ├─ no login wall crossing without user action
   └─ no upstream mutation from this ledger
```

## Verification

- Fresh official-source check on 2026-06-09 reached `https://www.browser-harness.com/` with HTTP 200.
- Fresh canonical GitHub check on 2026-06-09 reached `https://github.com/browser-use/browser-harness` with HTTP 200.
- GitHub tree check was non-truncated and included the required setup, skill, package, helper, and test paths.
- Upstream docs were read from raw URLs pinned to `6d20866664ea3d9691b27bbf64f42ae097437dc3`.
- Local clone advisory state: `/Users/yu/Projects/browser-harness` HEAD matches upstream HEAD and has no dirty entries.

## Remaining Risks

- Official upstream can change later; rerun the freshness check before using this as current doctrine.
- This ledger does not prove Browser Use cloud, profile sync, API-key, or billing behavior.
- This ledger does not authorize writes to external websites or changes to the upstream Browser Harness repository.

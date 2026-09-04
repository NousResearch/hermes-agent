# Hermes Agent 온보딩

새 팀원용 안내서. 분석 기준일 2026-09-04 · 커밋 `63279301bc` (main)

더 깊은 내용은 세 문서로 나뉘어 있습니다.

- [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) — 전체 구조 상세
- [RUN_AGENT_TRACE.md](RUN_AGENT_TRACE.md) — 실행 흐름 8단계 추적
- [CORE_FILES_RANKING.md](CORE_FILES_RANKING.md) — 핵심 파일 랭킹 근거

---

## 1. 프로젝트 개요

**Nous Research가 만든 오픈소스 개인 AI 에이전트.** 하나의 파이썬 에이전트 코어를 CLI, 메시징 게이트웨이(Telegram·Discord·Slack 등 22종), TUI, Electron 데스크톱 앱이 공유합니다.

차별점은 **닫힌 학습 루프**입니다. 경험에서 스킬을 자동 생성하고, 사용 중에 스킬을 스스로 개선하고, 메모리를 큐레이팅하고, 과거 대화를 FTS5로 검색합니다. 모델은 40개 프로바이더 플러그인으로 교체 가능하고, 로컬부터 서버리스까지 7종 터미널 백엔드에서 돕니다.

라이선스 MIT · 파이썬 파일 5,128개 · 테스트 파일 3,794개.

---

## 2. 먼저 알아야 할 두 원칙

코드를 읽기 전에 이것부터 알아야 합니다. **리뷰에서 PR이 반려되는 대부분의 이유가 이 둘입니다.**

### 원칙 1 — 대화별 프롬프트 캐싱은 신성불가침

장수 대화는 매 턴 캐시된 프리픽스를 재사용합니다. 과거 컨텍스트를 변형하거나, 툴셋을 교체하거나, 대화 도중 시스템 프롬프트를 재구성하면 캐시가 깨지고 사용자 비용이 배로 뜁니다. **유일한 예외가 컨텍스트 압축입니다.**

실제 적용 사례: 스킬 슬래시 커맨드는 시스템 프롬프트가 아니라 **user 메시지로 주입**됩니다(`agent/skill_commands.py`). 시스템 프롬프트에 넣으면 캐시가 깨지기 때문입니다.

### 원칙 2 — 코어는 좁은 허리, 기능은 가장자리에

모델 툴은 매 API 호출마다 스키마가 전송됩니다. 그래서 새 *코어* 툴의 기준선이 매우 높습니다.

새 기능을 넣을 때의 우선순위 (**Footprint Ladder**):

1. 기존 코드 확장
2. CLI 커맨드 + 스킬
3. 서비스로 게이팅된 툴 (`check_fn`)
4. 플러그인
5. 카탈로그의 MCP 서버
6. 새 코어 툴 (최후의 수단)

오해하지 마세요 — 제품 자체는 공격적으로 확장합니다. 새 플랫폼 어댑터, 프로바이더, 데스크톱/TUI 기능은 크더라도 환영이고 정기적으로 머지됩니다. 절제는 **코어 에이전트와 모델 툴 스키마**에만 적용됩니다. *가장자리는 확장적으로, 허리는 보수적으로.*

### 참고: 환영받는 작업

- **실제 버그를 제대로 고치기.** 머지되는 대부분이 `fix(...)`입니다. 좋은 수정은 현재 `main`에서 증상을 재현하고, 발현 지점을 정확히 짚고, 형제 호출 경로까지 포함해 **버그 클래스 전체**를 고칩니다.
- **god-file 리팩터링.** `cli.py` / `run_agent.py` / `gateway/run.py`에서 수천 줄 덩어리를 모듈로 추출하는 건 환영받는 작업입니다. diff가 크고 기계적이어도 괜찮습니다.

---

## 3. 디렉터리 구조

### 코어

| 경로 | 역할 |
| --- | --- |
| `run_agent.py` | `AIAgent` 클래스 — 에이전트 상태 소유자 |
| `agent/` | 에이전트 내부 — 대화 루프, 프로바이더 어댑터, 메모리, 압축, 큐레이터 |
| `model_tools.py` | 툴 오케스트레이션, `handle_function_call()` |
| `toolsets.py` | 툴셋 30종 정의, `_HERMES_CORE_TOOLS` |
| `tools/` | 툴 구현체 — `tools/registry.py`로 자동 발견 |
| `tools/environments/` | 터미널 백엔드 7종 (local, docker, ssh, modal, daytona, singularity, vercel_sandbox) |
| `hermes_state.py` | `SessionDB` — SQLite 세션 스토어 |
| `hermes_constants.py` | `get_hermes_home()` — 프로필 인식 경로 |

### 사용자 인터페이스

| 경로 | 역할 |
| --- | --- |
| `hermes_cli/` | CLI 서브커맨드, 셋업 마법사, 플러그인 로더, 스킨 엔진 |
| `cli.py` | `HermesCLI` — 인터랙티브 CLI 오케스트레이터 |
| `ui-tui/` | Ink(React) 터미널 UI — `hermes --tui` |
| `tui_gateway/` | TUI용 파이썬 JSON-RPC 백엔드 |
| `apps/desktop/` | Electron 데스크톱 앱 (독립 채팅 표면) |
| `web/` | Vite 대시보드 — xterm.js로 실제 TUI를 PTY 임베드 |
| `gateway/` | 메시징 게이트웨이 — `run.py` + `session.py` + `platforms/` |
| `acp_adapter/` | ACP 서버 (VS Code / Zed / JetBrains) |

### 확장

| 경로 | 역할 |
| --- | --- |
| `plugins/model-providers/` | 모델 프로바이더 40종 |
| `plugins/platforms/` | 메시징 플랫폼 22종 |
| `plugins/memory/` | 메모리 프로바이더 10종 (honcho, mem0, supermemory 등) |
| `skills/` | 기본 탑재 스킬 |
| `optional-skills/` | 무거운·틈새 스킬 — `hermes skills install`로 명시적 설치 |
| `cron/` | 스케줄러 — `jobs.py`, `scheduler.py` |

### 기타

| 경로 | 역할 |
| --- | --- |
| `tests/` | Pytest 스위트 |
| `scripts/` | `run_tests.sh`, `release.py` 등 |
| `website/` | Docusaurus 문서 사이트 |
| `AGENTS.md` | 96KB짜리 개발 규범 — 기여 루브릭, 함정, 테스트 안티패턴 |

**사용자 데이터 위치:** `~/.hermes/config.yaml`(설정), `~/.hermes/.env`(API 키만), `~/.hermes/logs/`(로그)

---

## 4. 핵심 파일 5개

중요도 순. 기준은 파일 크기가 아니라 **레버리지** — 한 줄을 바꿨을 때 시스템의 몇 퍼센트가 따라 바뀌는가.

| 순위 | 파일 | LOC | 역할 |
| --- | --- | --- | --- |
| 1 | `agent/conversation_loop.py` | 9,339 | 한 번의 유저 턴을 끝까지 굴리는 실제 엔진 |
| 2 | `run_agent.py` | 10,175 | `AIAgent` 클래스 — 에이전트의 모든 상태를 소유 |
| 3 | `model_tools.py` | 1,707 | `handle_function_call()` — 모든 툴 호출의 단일 관문 |
| 4 | `hermes_state.py` | 17,370 | `SessionDB` — WAL + FTS5 SQLite 세션 스토어 |
| 5 | `agent/context_compressor.py` | 9,246 | 컨텍스트 압축 |

**꼭 알아둘 것:** 1위와 2위는 "행위 소유자 vs 상태 소유자"로 쪼개져 있습니다. `AIAgent.run_conversation`(`run_agent.py:9272`)은 **얇은 포워더**이고, 실제 ~3,900줄짜리 루프 본문은 `agent/conversation_loop.py`에 있습니다. 추출된 함수들은 전부 부모 `AIAgent`를 첫 인자 `agent`로 받아 속성 조회로 상태에 접근합니다.

`AGENTS.md`의 "The core loop is inside `run_conversation()`" 서술은 이 추출을 반영하지 못한 상태입니다. **루프를 읽으려면 `agent/conversation_loop.py`를 여세요.**

랭킹 근거 전문은 [CORE_FILES_RANKING.md](CORE_FILES_RANKING.md)에 있습니다.

---

## 5. 주요 코드 흐름

### 5-1. 파일 의존성 사슬

```
tools/registry.py  (의존성 0 — 모든 툴 파일이 import)
       ↑ 각 tools/*.py가 import 시점에 register() 호출 — 자기등록
model_tools.py  (registry import + 툴 디스커버리 트리거)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

**함정:** `discover_plugins()`는 `model_tools.py` import의 부수효과로만 실행됩니다. `model_tools.py`를 먼저 import하지 않고 플러그인 상태를 읽는 코드는 `discover_plugins()`를 명시적으로 호출해야 합니다(멱등).

### 5-2. 한 턴이 도는 흐름

```
사용자 입력
│
├─ 진입점 (hermes → hermes_cli/main.py:13526)
│    └─ AIAgent(...) 생성
│
└─ agent.run_conversation(message)
     │
     ├─ 포워더 (run_agent.py:9272)
     │    ├─ 백그라운드 리뷰 선점
     │    ├─ durable turn lease — state.db 프로세스 간 직렬화 (fail-closed)
     │    └─ relay/accounting/portal 컨텍스트 세팅
     │
     ├─ 턴 프롤로그: build_turn_context()  (agent/turn_context.py:572)
     │    ├─ 시스템 프롬프트 restore-or-build ← 원칙 1이 여기서 지켜짐
     │    ├─ preflight 컨텍스트 압축
     │    ├─ pre_llm_call 플러그인 훅
     │    └─ 외부 메모리 prefetch
     │
     ├─ 메인 루프  (agent/conversation_loop.py:2289)
     │    while (예산 남음) or (1턴 유예 호출):
     │      ├─ 턴 중 유저 개입 흡수 (_drain_pending_redirect)
     │      ├─ API 호출
     │      ├─ finish_reason 분기: tool_calls / stop / length
     │      ├─ 도구 실행 → 세그먼트 플래너
     │      │     ├─ 1개 이하 → sequential
     │      │     ├─ 전부 병렬 안전 → concurrent
     │      │     └─ 혼합 → segmented (안전 구간만 병렬, 순서 보존)
     │      │           → agent/tool_executor.py
     │      │           → model_tools.handle_function_call()
     │      └─ 회복: 드롭된 tool-call, Codex ack, 압축 재시도
     │
     └─ finalize_turn()  (agent/turn_finalizer.py:138)
          ├─ _save_trajectory
          ├─ _persist_session  → SessionDB
          └─ 메모리/스킬 리뷰 넛지 ← 학습 루프가 여기서 발화
```

단계별 상세와 회복 분기는 [RUN_AGENT_TRACE.md](RUN_AGENT_TRACE.md)에 있습니다.

### 5-3. 새 툴을 추가할 때

**대부분의 경우 코어를 건드리면 안 됩니다.** 원칙 2의 Footprint Ladder를 먼저 확인하세요. 로컬 전용 툴이면 플러그인 경로를 씁니다.

```
~/.hermes/plugins/<name>/plugin.yaml
~/.hermes/plugins/<name>/__init__.py   → ctx.register_tool(...)
```

플러그인 툴셋은 자동 발견되고, `tools/`나 `toolsets.py`를 건드리지 않고 켜고 끌 수 있습니다.

### 5-4. 슬래시 커맨드를 추가할 때

`hermes_cli/commands.py`의 `COMMAND_REGISTRY`에 `CommandDef` 하나를 추가하면 **모든 하위 소비자가 자동 파생**됩니다 — CLI 디스패치, 게이트웨이 디스패치, `/help` 출력, Telegram BotCommand 메뉴, Slack 서브커맨드 매핑, 자동완성.

별칭만 추가하려면 기존 `CommandDef`의 `aliases` 튜플만 건드리면 됩니다.

---

## 6. 개발 시작하기

### 6-1. 설치 (권장 경로)

표준 인스톨러를 쓴 뒤, 인스톨러가 만든 git 체크아웃에서 작업합니다. 이 레이아웃이 `hermes update`, 관리 venv, 지연 의존성, 게이트웨이, 문서 도구가 전제하는 구조입니다.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Windows(네이티브)는 PowerShell에서:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

### 6-2. 설치 (수동 클론 — CI/일회성용)

**venv를 소스 트리 밖에 만드세요.** 에이전트가 자기 체크아웃에 대해 상대 경로 명령을 실행하면 트리 안의 venv를 통째로 날려서 실행 중인 런타임을 죽일 수 있습니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

**버전 요구사항:** Python `>=3.11,<3.14` (상한은 장식이 아님 — 3.14는 pydantic-core에 cp314 휠이 없어 소스 빌드로 폴백해 실패), Node `^22.22.0 || ^24.11.0 || >=26.0.0`

### 6-3. 실행

```bash
hermes                # 인터랙티브 CLI
hermes --tui          # Ink 터미널 UI
hermes model          # 프로바이더/모델 선택
hermes tools          # 툴 켜고 끄기 (curses UI)
hermes setup          # 전체 셋업 마법사
hermes gateway        # 메시징 게이트웨이
hermes doctor         # 문제 진단
```

디버깅용 독립 실행 (연구 경로 — 프로필·세션 복원을 거치지 않음):

```bash
python run_agent.py --query="..." --max_turns=10
python run_agent.py --list_tools
```

### 6-4. 테스트 — 반드시 래퍼를 쓸 것

**`pytest`를 직접 부르지 마세요.** `scripts/run_tests.sh`가 CI와의 환경 패리티를 강제합니다.

```bash
scripts/run_tests.sh                                     # 전체
scripts/run_tests.sh tests/gateway/                      # 디렉터리 하나
scripts/run_tests.sh tests/agent/test_foo.py -k test_x   # 파일 + 패턴
scripts/run_tests.sh -j 4                                # 병렬도 제한
scripts/run_tests.sh -v --tb=long                        # pytest 플래그 통과
```

래퍼가 강제하는 것:

| | 래퍼 없이 | 래퍼로 |
| --- | --- | --- |
| 프로바이더 API 키 | 내 환경에 있는 것 (풀 자동 감지) | 소수 예외 빼고 전부 unset |
| `HOME` / `~/.hermes/` | 실제 설정 + auth.json | 테스트마다 임시 디렉터리 |
| 타임존 | 로컬 (KST 등) | UTC |
| 로케일 | 환경값 | C.UTF-8 |

파일마다 **새로 스폰된 서브프로세스**에서 돌기 때문에 모듈 레벨 dict/set과 ContextVar가 파일 간에 새지 않습니다. 16코어 개발 머신에서 API 키를 켜둔 채 직접 `pytest`를 돌리면 CI와 갈라져서 "로컬은 되는데 CI가 깨짐"(그리고 그 반대)이 반복적으로 발생했습니다.

**Flake 정책:** 실패한 테스트 *파일*은 새 서브프로세스에서 1회 자동 재시도됩니다. 재시도 통과는 green으로 치되 `⚠ FLAKY` 섹션에 두 번의 출력이 함께 찍힙니다. **FLAKY 리포트는 무시할 노이즈가 아니라 고쳐야 할 버그입니다.**

### 6-5. JS/TS 테스트

워크스페이스별로 돌립니다.

```bash
cd ui-tui && npm run check     # build:ink + typecheck + test + lint
cd web     && npm run check    # typecheck + test + lint
cd apps/desktop && npx vitest run src/lib/desktop-slash-commands.test.ts
```

TUI 개발 중에는:

```bash
cd ui-tui
npm install     # 최초 1회
npm run dev     # watch 모드
```

---

## 7. 첫날 밟기 쉬운 지뢰

**`.venv`가 체크아웃에 없습니다.** `scripts/run_tests.sh`는 `.venv` → `venv` → `$HOME/.hermes/hermes-agent/venv` 순으로 탐색하되, **pytest가 실제로 설치된** venv만 고릅니다. 릴리스 venv에는 pytest가 없어서 건너뜁니다.

**테스트를 잘못된 언어 스위트에 두지 마세요.** `package.json`, `tsconfig.json`, `.ts`/`.tsx` 내용을 검사하는 테스트는 파이썬이 아니라 vitest 쪽에 있어야 합니다. CI 변경 분류기가 JS 파일만 바뀐 PR에서는 파이썬 테스트를 돌리지 않아서, 회귀가 PR에서 green이고 `main`에서 red가 됩니다.

**호스트 OS를 가짜로 만들지 마세요.** `sys.platform`을 패치하는 대신 `@pytest.mark.linux_only` / `macos_only` / `windows_only` 마커를 씁니다. 맨 `skipif`를 쓰면 Linux에서 skip되고 Windows 레인에서는 import조차 안 되어 **아무 호스트에서도 안 돌면서 green으로 보입니다.**

**change-detector 테스트를 쓰지 마세요.** 모델 카탈로그 스냅샷, config 버전 리터럴, 열거 개수 assert는 정상적인 소스 업데이트마다 CI를 깨뜨립니다. 대신 배선·마이그레이션·불변식을 테스트하세요.

**대시보드에 채팅 UI를 다시 만들지 마세요.** 대시보드는 실제 `hermes --tui`를 PTY로 임베드합니다. 트랜스크립트·컴포저·터미널은 Ink 소유이고, Ink에 추가하면 대시보드에 자동으로 나타납니다. 사이드바·인스펙터 같은 보조 UI는 React로 만들어도 됩니다.

**의존성에 범위를 쓰지 마세요.** 모든 직접 의존성이 `==X.Y.Z`로 고정돼 있습니다. 2026-05-12 mistralai PyPI 웜 대응 정책입니다. 버전을 올릴 때는 `pyproject.toml` 핀을 바꾸고 `uv lock`으로 lockfile을 재생성합니다.

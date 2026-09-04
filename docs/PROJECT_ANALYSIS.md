# Hermes Agent — 프로젝트 구조 분석

> 분석 기준일: 2026-09-04 · 대상 커밋: `63279301bc` (main)
> 출처: `README.md`, `AGENTS.md`, `pyproject.toml`, 실제 디렉터리 트리

---

## 1. 무엇을 하는 프로젝트인가

**Nous Research가 만든 오픈소스 개인 AI 에이전트.** `pyproject.toml`의 한 줄 요약:

> *"The self-improving AI agent — creates skills from experience, improves them during use, and runs anywhere"*

기존 코딩/개인 에이전트의 세 가지 한계를 정면으로 겨냥한다.

| 문제 | Hermes의 답 |
| --- | --- |
| 세션이 끝나면 배운 걸 다 잊는다 | **닫힌 학습 루프** — 경험에서 스킬을 자동 생성하고, 사용 중에 스킬을 스스로 개선하고, 메모리를 큐레이팅하고, 과거 대화를 FTS5로 검색한다 |
| 노트북에 묶여 있다 | **어디서든 실행** — 로컬/Docker/SSH/Singularity/Modal/Daytona/Vercel Sandbox 7종 터미널 백엔드. Telegram으로 대화하면서 작업은 클라우드 VM에서 돌린다 |
| 특정 모델 벤더에 락인된다 | **40개 모델 프로바이더 플러그인** — `hermes model` 한 줄로 교체, 코드 변경 없음 |

라이선스 MIT. 파이썬 파일 5,128개, 테스트 파일 3,794개(~17k 테스트) 규모.

---

## 2. 아키텍처 — "하나의 코어, 여러 개의 프론트엔드"

```
            ┌─── cli.py (prompt_toolkit CLI, 22k LOC)
            ├─── ui-tui/ (Ink/React TUI) ──JSON-RPC/stdio── tui_gateway/
프론트엔드  ├─── apps/desktop/ (Electron + assistant-ui) ──WS/JSON-RPC──┐
            ├─── web/ (Vite 대시보드, xterm.js로 TUI를 PTY 임베드)     │
            ├─── gateway/ (Telegram·Discord·Slack·WhatsApp·Signal…)    │
            └─── acp_adapter/ (VS Code / Zed / JetBrains)              │
                              ↓                                        │
              ┌────────────────────────────────────────────────────────┘
              │
코어      run_agent.py :: AIAgent  ← 동기 tool-calling 루프 (10k LOC)
              │   agent/  — 프로바이더 어댑터, 압축, 캐싱, 메모리, 큐레이터
              ├─ model_tools.py    — 툴 오케스트레이션 / handle_function_call()
              ├─ toolsets.py       — 30개 툴셋 번들, _HERMES_CORE_TOOLS
              ├─ tools/            — 툴 구현체 (import 시 registry에 자기등록)
              │    └ environments/ — 터미널 백엔드 7종
              └─ hermes_state.py   — SQLite 세션 스토어 + FTS5 검색 (17k LOC)
```

핵심은 **동일한 `AIAgent` 코어가 6개 프론트엔드에 공유된다**는 점이다. CLI든 Telegram이든 Electron이든 대화 루프·툴·세션은 같은 파이썬 코어에서 돈다.

### 2.1 설계를 지배하는 두 원칙

`AGENTS.md` 서두가 모든 설계 결정과 코드 리뷰의 렌즈로 못박은 두 가지:

1. **"대화별 프롬프트 캐싱은 신성불가침이다."**
   장수 대화는 매 턴 캐시된 프리픽스를 재사용한다. 과거 컨텍스트를 변형하거나, 툴셋을 교체하거나, 대화 도중 시스템 프롬프트를 재구성하면 캐시가 깨지고 사용자 비용이 배로 뛴다. 유일한 예외가 컨텍스트 압축.
   → 그래서 스킬 슬래시 커맨드조차 시스템 프롬프트가 아니라 **user 메시지로 주입**한다 (`agent/skill_commands.py`).

2. **"코어는 좁은 허리(narrow waist), 기능은 가장자리에."**
   모델 툴은 매 API 호출마다 스키마가 전송되므로 새 *코어* 툴의 기준선이 매우 높다. 새 기능 대부분은 CLI 커맨드 + 스킬, 서비스로 게이팅된 툴, 또는 플러그인으로 가야 한다.
   → 메시징 플랫폼 22종과 모델 프로바이더 40종이 전부 플러그인으로 코어 밖에 나가 있는 이유.

### 2.2 파일 의존성 사슬

```
tools/registry.py  (의존성 0 — 모든 툴 파일이 import)
       ↑
tools/*.py  (각자 import 시점에 registry.register() 호출 — 자기등록)
       ↑
model_tools.py  (registry import + 툴 디스커버리 트리거)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

**함정:** `discover_plugins()`는 `model_tools.py` import의 부수효과로만 실행된다. `model_tools.py`를 먼저 import하지 않고 플러그인 상태를 읽는 코드 경로는 `discover_plugins()`를 명시적으로 호출해야 한다(멱등).

### 2.3 에이전트 루프 — "상태 소유자 vs 행위 소유자" 분리

**루프 본문은 `run_agent.py`에 없다.** `AIAgent.run_conversation`(`run_agent.py:9272`)은 얇은 포워더이고, 실제 ~3,900줄짜리 본문은 `agent/conversation_loop.py`로 추출되어 있다.

```
run_agent.py :: AIAgent            ← 상태 소유자 (~60개 생성자 파라미터)
   └ run_conversation()  ────┐
                             ↓ 얇은 포워더
     agent/conversation_loop.py :: run_conversation(agent, ...)   ← 행위 소유자
     agent/tool_executor.py   :: _execute_tool_calls_{sequential,concurrent}(agent, ...)
```

추출된 함수들은 전부 부모 `AIAgent`를 첫 인자 `agent`로 받아 속성 조회로 상태에 접근한다:

```python
def _maybe_inject_run_budget_wrapup(agent: Any, messages: List[Dict[str, Any]]) -> bool:
def _should_rearm_compression_budget(agent, ...)
def _restore_or_build_system_prompt(agent, system_message, conversation_history):
```

추출 모듈은 `_ra()` 헬퍼로 원래 `run_agent` 모듈을 되짚는다 — 프로덕션 코드와 테스트가 `run_agent._set_interrupt`, `handle_function_call`, `OpenAI` 같은 심볼을 직접 패치하기 때문에, 그 패치가 계속 먹히게 하려는 장치다.

> **주의:** `AGENTS.md`의 "The core loop is inside `run_conversation()`" 서술은 이 추출을 반영하지 못한 상태다. 루프를 읽으려면 `agent/conversation_loop.py`를 열어야 한다.

한 턴이 통과하는 단계: 모델 호출 → 툴 디스패치 → 재시도/페일오버 → 압축 → 턴 후 훅 → 백그라운드 메모리·스킬 리뷰 넛지. 각 단계의 실제 코드 경로와 회복 분기는 [RUN_AGENT_TRACE.md](RUN_AGENT_TRACE.md)의 Phase 4~7에 있다.

개념적 골격은 아래와 같다. 전부 동기(synchronous)이며 인터럽트 체크, 예산 추적, 1턴 유예 호출이 얹혀 있다.

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

메시지는 OpenAI 포맷(`system`/`user`/`assistant`/`tool`)을 따르고, reasoning 콘텐츠는 `assistant_msg["reasoning"]`에 보관한다.
실제 `AIAgent.__init__`는 파라미터가 약 60개(크리덴셜, 라우팅, 콜백, 세션 컨텍스트, 예산, 크리덴셜 풀 등).

---

## 3. 디렉터리 지도

| 경로 | 역할 |
| --- | --- |
| `run_agent.py` | `AIAgent` 클래스 — 에이전트 상태 소유자 (10,175 LOC). 루프 본문은 `agent/conversation_loop.py`로 추출됨 (§2.3) |
| `cli.py` | `HermesCLI` — 인터랙티브 CLI 오케스트레이터 (22,426 LOC) |
| `model_tools.py` | 툴 오케스트레이션, `discover_builtin_tools()`, `handle_function_call()` |
| `toolsets.py` | 툴셋 정의, `_HERMES_CORE_TOOLS` 리스트 |
| `hermes_state.py` | `SessionDB` — SQLite 세션 스토어 + FTS5 검색 (17,370 LOC) |
| `hermes_constants.py` | `get_hermes_home()` — 프로필 인식 경로 |
| `agent/` | 에이전트 내부 — 프로바이더 어댑터, 메모리, 캐싱, 압축, 큐레이터 |
| `hermes_cli/` | CLI 서브커맨드, 셋업 마법사, 플러그인 로더, 스킨 엔진 |
| `tools/` | 툴 구현체 — `tools/registry.py`로 자동 발견 |
| `tools/environments/` | 터미널 백엔드 (local, docker, ssh, modal, daytona, singularity, vercel_sandbox) |
| `gateway/` | 메시징 게이트웨이 — `run.py` + `session.py` + `platforms/` |
| `plugins/` | 플러그인 시스템 (아래 §4.5) |
| `skills/` / `optional-skills/` | 기본 탑재 스킬 / 무거운·틈새 스킬(명시적 설치) |
| `ui-tui/` | Ink(React) 터미널 UI — `hermes --tui` |
| `tui_gateway/` | TUI용 파이썬 JSON-RPC 백엔드 |
| `apps/desktop/` | Electron 데스크톱 앱 (독립 채팅 표면) |
| `web/` | Vite 대시보드 (xterm.js로 실제 TUI를 PTY 임베드) |
| `acp_adapter/` | ACP 서버 (VS Code / Zed / JetBrains 연동) |
| `cron/` | 스케줄러 — `jobs.py`, `scheduler.py` |
| `website/` | Docusaurus 문서 사이트 |
| `tests/` | Pytest 스위트 (~3,794 파일) |

**사용자 설정:** `~/.hermes/config.yaml` (설정), `~/.hermes/.env` (API 키 전용)
**로그:** `~/.hermes/logs/` — `agent.log`(INFO+), `errors.log`(WARNING+), `gateway.log`

---

## 4. 주요 서브시스템

### 4.1 학습 루프 (이 프로젝트의 차별점)

- **스킬 3계층** — `skills/`(기본 탑재) + `optional-skills/`(무거운/틈새, `hermes skills install`로 명시적 설치) + `~/.hermes/skills/`(에이전트가 만든 것). [agentskills.io](https://agentskills.io) 오픈 표준 호환.
- **큐레이터** (`agent/curator.py`) — 에이전트 생성 스킬의 사용 통계를 추적해 stale → archive로 자동 전이. 불변식이 엄격하다:
  - `created_by: "agent"` 프로비넌스인 스킬만 건드린다. 번들·허브 설치 스킬은 손대지 않는다.
  - **절대 삭제하지 않는다.** 최대 파괴적 행동이 아카이브(`~/.hermes/skills/.archive/`, 복원 가능).
  - pin된 스킬은 모든 자동 전이와 LLM 리뷰 패스에서 면제.
- **메모리** — `agent/memory_manager.py` + `plugins/memory/` 10종 (honcho, mem0, supermemory, retaindb, byterover, hindsight, holographic, openviking …). Honcho는 dialectic 사용자 모델링.
- **세션 검색** — `tools/session_search_tool.py` + FTS5 + LLM 요약으로 크로스 세션 리콜.

### 4.2 툴 시스템

`tools/registry.py`가 의존성 0인 최하위 레이어이고, 각 `tools/*.py`가 import 시점에 `register()`를 호출하는 **자기등록** 방식.

툴셋 30종으로 번들링되어 플랫폼별로 다른 조합을 쓴다(예: Telegram은 `messaging` 베이스):

`browser`, `clarify`, `code_execution`, `cronjob`, `debugging`, `delegation`, `discord`, `discord_admin`, `feishu_doc`, `feishu_drive`, `file`, `homeassistant`, `image_gen`, `kanban`, `memory`, `messaging`, `moa`, `rl`, `safe`, `search`, `session_search`, `skills`, `spotify`, `terminal`, `todo`, `tts`, `video`, `vision`, `web`, `yuanbao`

설정은 `hermes tools`(curses UI) 또는 `config.yaml`의 `tools.<platform>.enabled/disabled`.

### 4.3 위임 / 병렬화 (`delegate_task`)

`tools/delegate_tool.py`가 격리된 컨텍스트 + 터미널 세션을 가진 서브에이전트를 스폰한다.

- **형태:** 단일(`goal`) 또는 배치 병렬(`tasks: [...]`, 동시성 상한 `delegation.max_concurrent_children`, 기본 3)
- **역할:** `leaf`(기본, 재위임·clarify·memory·send_message·cronjob 불가, `execute_code`는 유지) vs `orchestrator`(재위임 가능, `max_spawn_depth` 기본 2로 제한)
- `background=true`면 위임 ID를 즉시 반환하고, 결과는 비동기 완료 큐를 통해 나중에 대화에 재진입
- **내구성 규칙:** background 위임은 턴에서 분리되지만 여전히 프로세스 로컬이다. 프로세스 재시작을 살아남아야 하는 작업은 `cronjob` 또는 `terminal(background=True, notify_on_complete=True)`를 써야 한다.

### 4.4 Kanban (멀티에이전트 작업 큐)

SQLite 기반 durable 보드. 디스패처(기본 60초 주기)가 stale claim 회수 → ready 태스크 승격 → 원자적 claim → 배정된 프로필 스폰을 반복한다. 기본값 `kanban.dispatch_in_gateway: true`로 게이트웨이 안에서 돈다.

- **격리 모델:** board = 하드 경계(워커 env에 `HERMES_KANBAN_BOARD` 고정 → 다른 보드가 안 보임), tenant = 보드 *내부*의 소프트 네임스페이스(워크스페이스 경로 + 메모리 키 격리)
- 같은 태스크에서 `kanban.failure_limit`(기본 2)회 연속 실패하면 디스패처가 자동 차단해 스핀 루프를 막는다

### 4.5 플러그인

`register(ctx)` 함수 하나로 세 가지를 등록한다:

- **라이프사이클 훅** — `pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`
- **툴** — `ctx.register_tool(...)`
- **CLI 서브커맨드** — `ctx.register_cli_command(...)`. 플러그인의 argparse 트리가 시작 시 `hermes`에 배선되어 `main.py` 변경 없이 `hermes <pluginname> <subcmd>`가 동작

디스커버리 소스: `~/.hermes/plugins/`, `./.hermes/plugins/`, pip entry points, 그리고 레포의 `plugins/`.

레포에 실린 플러그인 카테고리:

| 카테고리 | 내용 |
| --- | --- |
| `plugins/model-providers/` (40종) | anthropic, openrouter, gemini, bedrock, vertex, xai, deepseek, ollama-cloud, nous, copilot, openai-codex, qwen-oauth … |
| `plugins/platforms/` (22종) | telegram, discord, slack, whatsapp, matrix, teams, feishu, wecom, dingtalk, line, irc, sms, email, ntfy, a2a … |
| `plugins/memory/` (10종) | honcho, mem0, supermemory, retaindb, byterover, hindsight, holographic, openviking |
| 기타 | kanban, observability, image_gen, video_gen, browser, spotify, google_meet, disk-cleanup, hermes-achievements, security-guidance, teams_pipeline |

> 참고: `gateway/platforms/`에도 어댑터가 일부 남아 있다(api_server, webhook, signal, whatsapp_cloud, bluebubbles, weixin, yuanbao, qqbot). 신규 플랫폼 추가 가이드는 `gateway/platforms/ADDING_A_PLATFORM.md`.

### 4.6 Cron (스케줄 작업)

`cron/jobs.py`(잡 스토어) + `cron/scheduler.py`(틱 루프). 에이전트는 `cronjob` 툴로, 사용자는 `hermes cron <verb>` 또는 `/cron`으로 스케줄한다.

- **스케줄 포맷:** duration(`30m`, `2h`), "every" 구문(`every monday 9am`), 5-field cron(`0 9 * * *`), ISO 타임스탬프(1회성)
- **잡 필드:** `skills`, `model`/`provider` 오버라이드, `script`(사전 데이터 수집 스크립트 — stdout이 프롬프트에 주입, `no_agent=True`면 스크립트가 잡 전체), `context_from`(잡 A의 출력 → 잡 B의 프롬프트), `workdir`, 멀티플랫폼 전달
- **하드닝 불변식:** 크론 세션에 3분 하드 인터럽트(폭주 루프가 스케줄러를 독점 불가), 캐치업 윈도우는 주기의 절반을 120s–2h로 클램프, 1회성 잡 놓친 발화에 120s 유예, `~/.hermes/cron/.tick.lock` 파일 락으로 프로세스 간 중복 틱 방지
- 크론 세션은 기본 `skip_memory=True` — 메모리 프로바이더는 의도적으로 크론에서 돌지 않는다
- 크론 전달은 대상 게이트웨이 세션에 미러링되지 **않는다**. 메인 대화의 메시지 role 교대가 깨지지 않도록 자체 크론 세션에 헤더/푸터 프레임과 함께 착지한다

### 4.7 슬래시 커맨드 레지스트리

`hermes_cli/commands.py`의 단일 `COMMAND_REGISTRY`(`CommandDef` 리스트)에서 **모든 하위 소비자가 자동 파생**된다: CLI 디스패치, 게이트웨이 디스패치, `/help` 출력, Telegram BotCommand 메뉴, Slack `/hermes` 서브커맨드 매핑, 자동완성, 카테고리별 CLI 도움말.

→ 별칭 하나 추가하려면 기존 `CommandDef`의 `aliases` 튜플만 건드리면 된다. 나머지는 전부 자동 갱신.

---

## 5. UI 표면의 경계 규칙

`AGENTS.md`가 명시적으로 못박은 "채팅 표면 중복 금지" 규칙:

- **대시보드(`web/`)는 실제 `hermes --tui`를 PTY로 임베드한다.** 리라이트가 아니다. `hermes_cli/pty_bridge.py` + `web_server.py`의 `@app.websocket("/api/pty")` → 브라우저의 xterm.js(WebGL 렌더러). 메인 트랜스크립트·컴포저·터미널은 Ink 소유이므로, **React로 재구현하지 말고 Ink를 확장하라.** Ink에 추가한 것은 대시보드에 자동으로 나타난다.
- 다만 **TUI를 둘러싼 구조화된 React UI는 허용** — 사이드바 위젯, 인스펙터, 요약, 상태 패널(`ChatSidebar`, `ModelPickerDialog`, `ToolCall` 등)처럼 트랜스크립트를 대체하지 않고 보완하는 것.
- **Electron 데스크톱(`apps/desktop/`)만 예외적으로 독립 표면.** `hermes --tui`를 임베드하지 않고 자체 컴포저·트랜스크립트·슬래시 파이프라인을 갖는다. headless `hermes serve` 백엔드를 스폰하며, WS/JSON-RPC 트랜스포트는 `apps/shared`(`@hermes/shared`)에 있어 대시보드와 공유한다. 데스크톱은 대시보드 프론트엔드에 빌드/런타임 의존이 없다.

---

## 6. 눈에 띄는 엔지니어링 관행

### 의존성 exact pin 정책

모든 직접 의존성이 `==X.Y.Z`로 정확히 고정되어 있고 범위 지정이 금지되어 있다. `pyproject.toml`에 이유가 주석으로 박혀 있다 — **2026-05-12 mistralai 2.4.6을 덮친 Mini Shai-Hulud PyPI 웜** 대응. 범위였다면 격리 전 몇 시간 동안 모든 설치가 감염 버전을 끌어왔을 것이다.

여기에 **스코프 규칙**이 붙는다: *모든* Hermes 세션이 쓰는 패키지만 코어 `dependencies`에 둔다. 프로바이더 종속적인 것(`anthropic`, `firecrawl-py`, `exa-py`, `fal-client`, `edge-tts`, `parallel-web`)은 extra로 빼고 `tools/lazy_deps.py`가 사용자가 그 백엔드를 고를 때 지연 설치한다. **코어 deps가 작을수록 다음 공급망 공격의 폭발 반경이 작다.**

`requires-python = ">=3.11,<3.14"`의 상한도 장식이 아니다 — 상속된 `UV_PYTHON`이 3.14를 고르면 Rust 백엔드 트랜지티브(pydantic-core)에 cp314 휠이 없어 maturin 소스 빌드로 폴백해 실패한다. 상한이 있으면 uv가 시도 대신 명확한 에러를 낸다.

### 문서 중심 규범

`AGENTS.md`가 96KB. 기여 루브릭("The Footprint Ladder"), TypeScript 스타일 가이드(nanostore 우선, god hook 금지, 조건 사다리 대신 테이블 주도), 스킬 저작 표준(HARDLINE — `description` 60자 이하, 프로즈에서 셸 유틸 대신 네이티브 툴 이름 사용, `platforms:` 게이팅을 실제 import에 대해 감사), 알려진 함정, 테스트 안티패턴(모델 카탈로그 스냅샷 assert 금지, config 버전 리터럴 assert 금지 — 대신 배선/마이그레이션/불변식을 테스트하라)까지 전부 명문화되어 있다.

### 기타

- **프로필 기반 멀티 인스턴스** — `get_hermes_home()`이 경로를 프로필 인식으로 해석
- 초대형 모듈이 존재한다: `cli.py` 22k LOC, `hermes_state.py` 17k LOC, `run_agent.py` 10k LOC, `agent/conversation_loop.py` 9.3k LOC, `agent/context_compressor.py` 9.2k LOC
- **연구용 출력** — `batch_runner.py`(배치 트라젝토리 생성), `trajectory_compressor.py`(차세대 tool-calling 모델 학습용 압축), `mini_swe_runner.py`, `evals/`

---

## 7. 핵심 비즈니스 로직 파일 5개

중요도 순. 랭킹 기준은 **파일 크기가 아니라 레버리지** — 여기 한 줄을 바꿨을 때 시스템의 몇 퍼센트가 따라 바뀌는가.

| 순위 | 파일 | LOC | 역할 |
| --- | --- | --- | --- |
| **1** | `agent/conversation_loop.py` | 9,339 | 한 번의 유저 턴을 끝까지 굴리는 실제 엔진 — 모델 호출, 툴 디스패치, 재시도/페일오버, 압축, 턴 후 훅, 메모리·스킬 리뷰 넛지 |
| **2** | `run_agent.py` | 10,175 | `AIAgent` 클래스 본체 — 에이전트의 모든 상태(~60개 생성자 파라미터: 크리덴셜, 라우팅, 예산, 세션, 콜백)를 소유하는 객체 |
| **3** | `model_tools.py` | 1,707 | `handle_function_call()` — 모든 툴 호출이 반드시 통과하는 단일 관문(플러그인 훅, 미들웨어, 관측성, 에러 분류) |
| **4** | `hermes_state.py` | 17,370 | `SessionDB` — WAL + FTS5 SQLite 세션 스토어, 대화가 프로세스 밖에서 살아남는 유일한 지점 |
| **5** | `agent/context_compressor.py` | 9,246 | 컨텍스트 압축 — 대화가 컨텍스트 윈도우를 넘어서도 계속되게 만드는 로직 |

**차점자:** `toolsets.py`(1,062), `agent/tool_executor.py`(2,940), `tools/registry.py`(1,372), `agent/system_prompt.py`(1,172)

각 순위의 근거 — 1위와 2위를 가른 "상태 소유자 vs 행위 소유자" 기준, 3위가 1,700줄인데도 3위인 이유, 4위가 절대량 1등인데 4위인 이유, 5위로 `toolsets.py` 대신 압축을 고른 이유 — 는 [CORE_FILES_RANKING.md](CORE_FILES_RANKING.md)에 있다.

---

## 8. 더 파볼 만한 지점

| 관심사 | 시작점 |
| --- | --- |
| 에이전트 루프 내부 | `agent/conversation_loop.py` → `agent/tool_executor.py`, `agent/turn_finalizer.py` |
| 툴 등록·디스패치 경로 | `tools/registry.py` → `model_tools.py::handle_function_call` → `agent/tool_executor.py` |
| 학습 루프 구현 | `agent/curator.py`, `agent/learn_prompt.py`, `agent/learning_graph.py`, `tools/skill_usage.py` |
| 컨텍스트 압축 / 캐싱 | `agent/context_compressor.py`, `agent/prompt_caching.py`, `agent/prompt_cache_boundary.py` |
| 게이트웨이 세션 관리 | `gateway/run.py`, `gateway/session.py`, `gateway/turn_lease.py` |
| 프로바이더 어댑터 패턴 | `agent/anthropic_adapter.py`, `agent/bedrock_adapter.py`, `agent/vertex_adapter.py`, `plugins/model-providers/` |

---

## 9. 진입점(Entry Point)

### 콘솔 스크립트 (`pyproject.toml:391`)

```toml
[project.scripts]
hermes       = "hermes_cli.main:main"      # 주 진입점
hermes-agent = "run_agent:main"            # 에이전트 직접 실행 (fire CLI)
hermes-acp   = "acp_adapter.entry:main"    # ACP 서버 (VS Code/Zed/JetBrains)
```

레포 루트의 `./hermes` 셸 래퍼도 `hermes_cli.main:main`을 호출한다.

`python run_agent.py`를 직접 실행했을 때의 전체 코드 흐름(모듈 임포트 부수효과 → `fire` CLI 파싱 → `AIAgent` 생성 → 턴 프롤로그 → 메인 루프 → finalize)은 [RUN_AGENT_TRACE.md](RUN_AGENT_TRACE.md)에 8단계로 추적해 뒀다. 그 경로는 연구·디버깅용이라 프로필·세션 복원·스킬 로딩을 거치지 않지만, `run_conversation()` 진입 시점부터는 `hermes` 경로와 합류하므로 실사용 흐름을 읽을 때도 그 문서의 Phase 4 이후가 그대로 적용된다.

> **이 프로젝트는 import해서 쓰는 라이브러리가 아니다.** `setup.py`가 Nix 빌드 밖에서의 wheel/sdist 생성을 의도적으로 차단한다(`_GuardedBdistWheel`, `_GuardedSdist`). wheel이 번들 에셋(locales, skills, optional-mcps, web_dist, tui_dist, 플러그인 매니페스트) 없이 나가기 때문 — 이것들은 런타임에 env-var 오버라이드나 소스 체크아웃 레이아웃으로 해석된다. 개발은 editable install(`uv pip install -e .`)이 전제다.

### `hermes` 실행 시 코드가 도는 순서

`main()`은 `hermes_cli/main.py:13526`에 있지만, 거기 닿기 전에 import 부수효과로 도는 코드가 6단계 있다. 전부 "깨진 설치에서도 살아남기" 위한 방어 로직이다.

| # | 위치 | 하는 일 |
| --- | --- | --- |
| **1** | `hermes_cli/__init__.py:92` — `_ensure_utf8()` | **가장 먼저 실행되는 코드.** `hermes_cli.main`을 import하려면 패키지 `__init__.py`가 먼저 돌기 때문 |
| 2 | `main.py:60` — `import hermes_bootstrap` | Windows UTF-8 stdio (POSIX no-op) |
| 3 | `main.py:72` — `suppress_platform_ver_console()` | Windows 콘솔 창 깜빡임 억제 |
| 4 | `main.py:84` — `from hermes_cli import _startup_fast` | sys.path 부트스트랩 (스크립트 모드 대응) |
| 5 | `main.py:101` — `_early_recovery_mod.recover_if_needed()` | venv 자가 치유 |
| 6 | `main.py:128` — `_arm_sw()` | 게이트웨이 startup 워치독 무장 |
| 7 | `main.py:13526` — `main()` | 실제 진입점 |

**1단계가 최초인 이유** — `_ensure_utf8()`은 stdout/stderr가 UTF-8이 아니면 재구성한다. cp1252 Windows 서비스, latin-1/C 로케일 리눅스(최소 Debian, 라즈베리파이)에서 배선 문자(`┌│├└─`)와 `⚕` 글리프 출력이 `UnicodeEncodeError`를 던져 **커맨드가 시작도 못 하고 죽는 것**을 막는다.

`main.py:56` 주석은 "hermes_bootstrap must be the very first import"라고 하지만 엄밀히는 패키지 `__init__.py`가 먼저 돈다. 모순이 아니라 의도된 계층이고, `__init__.py` docstring이 직접 밝힌다 — **`__init__` = 플랫폼 무관 최초 방어선, `hermes_bootstrap` = Windows 전용 후속 레이어**(이미 복구된 스트림에 대해서는 멱등 no-op).

**2·5단계는 "부분 업데이트 생존" 장치**
- 2단계는 `ModuleNotFoundError`를 삼킨다. `hermes_bootstrap`은 `py-modules`로 등록된 최상위 모듈이라, `hermes update`가 `git reset --hard`와 `uv pip install -e .` 사이에서 죽으면 editable 설치의 `.pth`가 옛 모듈 목록을 가리킨다. 가드가 없으면 hermes가 import에서 크래시하고 **사용자는 복구하려고 `hermes update`를 실행하는 것조차 불가능**해진다.
- 5단계 `_early_recovery`는 stdlib만 쓴다(망가진 venv에서도 import 안전). 모듈 레벨 서드파티 import보다 **앞에** 있어야 하는 이유는, `main()` 안의 전체 복구 경로에 도달하기 전에 아래쪽 `from hermes_cli.env_loader import ...`가 먼저 크래시하기 때문이다.

**6단계는 argv 형태를 정확히 매칭**

```python
def _argv_is_gateway_run(argv: list) -> bool:
    return any(a == "gateway" and b == "run" for a, b in zip(argv, argv[1:]))
```

**인접한** 토큰 쌍만 매칭한다. `-p <profile>` 같은 전역 플래그가 끼어도 잡히지만, 두 단어를 다른 인자에서 우연히 언급하는 커맨드는 300초 하드 종료 타이머를 무장시키지 않는다. import 시점 데드락(네이티브 확장 초기화, import 락 경합)이 정확히 이 워치독의 대상이라 무거운 import 그래프 **이전에** 무장한다.

### 모듈을 직접 import할 때 — `model_tools`가 진짜 폭탄

`import model_tools`(또는 `import run_agent` — `run_agent.py:183`이 `model_tools`를 import한다)는 모듈 레벨 부수효과로 툴 디스커버리를 발화시킨다(`model_tools.py:230`):

```python
discover_builtin_tools()   # tools/*.py 전부 import → 각자 registry.register() 호출

try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()      # 사용자/프로젝트/pip 플러그인
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)
```

`discover_builtin_tools()`는 `tools/`의 모든 `*.py`를 glob해 import한다(`__init__.py`, `registry.py`, `mcp_tool.py` 제외). 파일당 AST 스캔이 ~100개 파일에 ~145ms라 `(mtime_ns, size)` 키로 디스크에 메모이즈한다.

**MCP 디스커버리는 여기서 제외됐다** (#16856). `discover_mcp_tools()`가 내부적으로 `future.result(timeout=120)`으로 블로킹하는데, 게이트웨이가 asyncio 이벤트 루프 안에서 첫 사용자 메시지에 이 모듈을 lazy-import하면서 **Discord/Telegram 하트비트를 최대 120초 얼려버렸기** 때문이다. 지금은 각 진입점이 자기 시작 시점에 명시적으로 실행한다 — `gateway/run.py`는 `run_in_executor`, `cli.py`는 인라인, `tui_gateway/server.py`는 인라인, `acp_adapter/server.py`는 `asyncio.to_thread`.

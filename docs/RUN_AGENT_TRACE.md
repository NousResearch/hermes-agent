# `python run_agent.py` 실행 흐름 추적

분석 기준일: 2026-09-04 · 대상 커밋: `63279301bc` (main)

## 전제: 이건 `hermes`가 타는 경로가 아니다

`run_agent.py`를 직접 실행하는 건 **연구·디버깅용 독립 진입점**입니다. 일반 사용자의 `hermes`는 `hermes_cli/main.py:main()`을 타고, 거기서 `AIAgent`를 생성합니다. 두 경로는 Phase 4부터 합류합니다.

- `hermes` → `hermes_cli/main.py:13526` → 설정/프로필/세션 해석 → `AIAgent(...)`
- `python run_agent.py` → `fire.Fire(main)` → 최소 파라미터로 `AIAgent(...)`

`run_agent.py`의 `main()`은 `max_turns=10`, OpenRouter 기본값 같은 연구용 프리셋을 씁니다. 프로필, 세션 복원, 스킬 로딩 같은 건 거치지 않습니다.

---

## Phase 0 — 프로세스 시작

파일 맨 끝이 진입점입니다.

```python
if __name__ == "__main__":
    import fire
    fire.Fire(main)
```

하지만 이 두 줄에 닿기 전에 **모듈 레벨 코드가 전부 먼저 실행**됩니다. 실제 비용의 대부분이 여기 있습니다.

---

## Phase 1 — 모듈 임포트 부수효과 (진짜 무거운 구간)

### 1-1. `hermes_bootstrap` (line 24)

```python
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass
```

Windows UTF-8 stdio 설정. POSIX에서는 no-op. `ModuleNotFoundError`를 삼키는 이유는 `hermes update`가 `git reset --hard`와 `uv pip install -e .` 사이에서 죽었을 때 editable 설치의 `.pth`가 옛 모듈 목록을 가리키기 때문 — 가드가 없으면 여기서 크래시해서 복구 명령조차 못 돌립니다.

### 1-2. 의도적으로 지연시킨 두 임포트

**`OpenAI` SDK는 모듈 top에 없습니다.** SDK가 ~240ms의 임포트를 끌고 오기 때문에, 첫 호출이나 `isinstance` 체크 때 SDK를 로드하는 얇은 프록시 객체로 노출합니다. 이 설계가 지키는 두 가지:

- `_create_openai_client`의 단일 `OpenAI(**client_kwargs)` 호출 지점
- 약 28개 테스트 파일이 쓰는 `patch("run_agent.OpenAI", ...)` 패턴

**`fire`도 `__main__` 블록 안에서만 임포트**합니다. 라이브러리로 쓸 때는 필요 없고, 데몬 스레드(예: 큐레이터의 fork된 리뷰 에이전트)에서 `run_agent`를 임포트할 때 `fire`가 없는 부분 설치에서 `ModuleNotFoundError`가 나지 않게 하려는 목적입니다.

### 1-3. `model_tools` 임포트 (line 183) — 폭탄

```python
from model_tools import (...)
```

`model_tools.py:230`의 모듈 레벨 부수효과가 여기서 발화합니다.

```python
discover_builtin_tools()   # tools/*.py 전부 import → 각자 registry.register() 호출

try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()      # 사용자/프로젝트/pip 플러그인
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)
```

`discover_builtin_tools()`는 `tools/`의 모든 `*.py`를 glob해 임포트합니다(`__init__.py`, `registry.py`, `mcp_tool.py` 제외). 파일당 AST 스캔이 ~100개 파일에 ~145ms라 `(mtime_ns, size)` 키로 디스크에 메모이즈합니다.

**MCP 디스커버리는 여기서 빠져 있습니다.** `discover_mcp_tools()`가 내부적으로 `future.result(timeout=120)`으로 블로킹하는데, 게이트웨이가 asyncio 이벤트 루프 안에서 이 모듈을 lazy-import하면서 Discord/Telegram 하트비트를 최대 120초 얼려버렸기 때문입니다. 각 진입점이 자기 시작 시점에 명시적으로 실행합니다.

`run_agent.py`는 `tools.terminal_tool`, `tools.interrupt`, `tools.browser_tool`도 직접 임포트합니다(line 189-191).

---

## Phase 2 — `fire.Fire(main)` — CLI 파싱

`main()`의 시그니처가 그대로 CLI 플래그가 됩니다 (`run_agent.py:9959`).

| 플래그 | 기본값 | 역할 |
| --- | --- | --- |
| `--query` | None | 자연어 질의. None이면 Python 3.13 예제 질의로 대체 |
| `--model` | `""` | 빈 문자열 → `__init__`에서 설정/프로바이더로 해석 |
| `--max_turns` | 10 | 툴 호출 이터레이션 상한 |
| `--enabled_toolsets` | None | 쉼표 구분 문자열 → 리스트로 파싱 |
| `--disabled_toolsets` | None | 쉼표 구분 문자열 → 리스트로 파싱 |
| `--list_tools` | False | 툴 목록만 출력하고 조기 반환 |
| `--save_trajectories` | False | `trajectory_samples.jsonl`에 append |
| `--save_sample` | False | UUID 이름의 단일 JSON 파일로 저장 |

`--list_tools`가 켜지면 툴셋/툴 목록을 출력하고 **`return`으로 끝납니다.** 에이전트는 생성되지 않습니다.

`AIAgent`의 `max_iterations` 기본값은 `sys.maxsize`(무제한)이지만, `main()`은 `max_turns=10`을 넘겨 연구용으로 묶어둡니다.

---

## Phase 3 — `AIAgent` 생성

```python
agent = AIAgent(
    base_url=base_url, model=model, api_key=api_key,
    max_iterations=max_turns,
    enabled_toolsets=enabled_toolsets_list,
    disabled_toolsets=disabled_toolsets_list,
    save_trajectories=save_trajectories,
    verbose_logging=verbose, log_prefix_chars=log_prefix_chars,
)
```

`RuntimeError`만 잡아서 `❌ Failed to initialize agent`로 출력하고 반환합니다. 다른 예외는 그대로 올라갑니다.

`AIAgent.__init__`(`run_agent.py:490`)은 파라미터가 약 60개입니다 — 크리덴셜, 라우팅(`providers_allowed`/`providers_ignored`/`providers_order`), API 모드(`chat_completions` / `codex_responses` / `codex_app_server`), 세션 컨텍스트, 예산, 콜백, 크리덴셜 풀. `main()`이 채우는 건 그중 9개뿐이고 나머지는 기본값으로 해석됩니다.

---

## Phase 4 — `run_conversation()` 진입 (여기서 `hermes` 경로와 합류)

```python
result = agent.run_conversation(user_query)
```

`AIAgent.run_conversation`(`run_agent.py:9272`)은 **얇은 포워더**입니다. 루프 본문은 `agent/conversation_loop.py`에 있습니다. 다만 포워더도 그냥 넘기지는 않고, 넘기기 전에 턴 경계 작업을 합니다.

### 4-1. 백그라운드 리뷰 선점

```python
cancel_background_review_for_live_turn(self)
```

백그라운드 리뷰는 프롬프트 캐시 패리티를 위해 같은 `session_id`를 공유합니다. 라이브 턴이 시작되면 리뷰 시작을 막거나 이미 들어간 요청을 인터럽트하고, 그 요청이 빠져나갈 때까지 기다린 뒤에야 같은 세션의 Relay/태스크 계측을 엽니다. 리뷰가 정해진 데드라인 안에 응답하지 않으면 포그라운드가 우선권을 유지합니다.

### 4-2. 턴 라이브니스 마킹

```python
_review_queue.note_turn_started()
```

지연 리뷰 큐가 두 개의 빠른 프롬프트 사이 틈으로 끼어들지 못하게 합니다.

### 4-3. Durable turn lease (`state.db` 프로세스 간 직렬화)

게이트웨이의 asyncio lease는 한 프로세스 안에서만 별칭 라우팅을 닫습니다. 이 durable lease는 Desktop, CLI resume, 게이트웨이, 백그라운드 전달 프로세스가 **같은 `state.db`를 공유할 때** load → run → flush 구간 전체를 직렬화합니다.

세션 행 존재 여부 확인이 예외로 실패하면 "세션 없음"이 아니라 **있는 것으로 간주하고 lease를 획득**합니다(fail-closed). 락 걸린 읽기가 실패했다고 행이 없다는 증거는 아니기 때문 — 예전에 fail-open으로 처리했다가 정확히 이 경합 지점에서 직렬화 없이 돌았습니다.

### 4-4. 컨텍스트 토큰 세팅

`relay_turn_id` 생성, accounting 컨텍스트, Portal affinity scope, 서브에이전트 부모 바인딩. 전부 `finally`에서 되돌립니다.

---

## Phase 5 — 턴 프롤로그: `build_turn_context()`

`agent/turn_context.py:572`. 한 턴에 한 번만 하는 설정이 전부 여기 모여 있고, `agent` 객체를 변형한 뒤 루프가 읽을 로컬들을 반환합니다.

수행 항목:

- stdio 가드 설치, 재시도 카운터 리셋
- 유저 메시지 sanitize (surrogate 제거)
- todo / nudge 하이드레이션
- **시스템 프롬프트 restore-or-build** — 세션별로 캐싱해서 프리픽스 캐싱을 지킴
- idle 트리거 압축 (opt-in, `idle_compact_after_seconds`)
- **preflight 컨텍스트 압축** — 보내기 전에 크기를 줄임
- `pre_llm_call` 플러그인 훅
- 외부 메모리 prefetch
- 크래시 대비 persist
- `api_content` 사이드카 — "보낸 것을 저장한다"

### 조기 종료 경로: `PreflightCompressionTimedOut`

압축이 호스트 타임아웃에 걸렸는데 요청이 여전히 오버사이즈면, **프로바이더 호출을 아예 보내지 않고** 타입 있는 복구 결과를 반환합니다.

```python
return {
    "final_response": _final_response,
    "messages": list(conversation_history or []),
    "completed": False,
    "api_calls": 0,
    "error": _final_response,
    "partial": True,
    "failed": True,
    "compression_exhausted": True,
    "turn_exit_reason": "context_compression_timeout",
}
```

일반 예외로 흘려보내지 않는 이유: 게이트웨이는 원시 예외 텍스트를 사용자에게 숨기기 때문에, 그러면 "`/compress` 실행 후 재시도" 같은 실행 가능한 안내가 묻히고 `compression_exhausted` 클린 세션 복구 계약도 건너뛰게 됩니다.

이 경로는 인바운드 유저 행을 **의도적으로 저장하지 않습니다.** 게이트웨이가 `compression_exhausted` 결과에 대해 트랜스크립트 저장을 건너뛰어 세션 비대화 루프를 막고, 자동 리셋이 이후 입력을 클린 세션으로 옮깁니다.

### 분기: `codex_app_server` 모드

`agent.api_mode == "codex_app_server"`면 여기서 턴 전체를 Codex 앱 서버 서브프로세스에 넘기고 **기본 Hermes 경로를 완전히 우회**합니다. 터미널·파일 조작·패치가 전부 Codex 안에서 돕니다.

---

## Phase 6 — 메인 루프

`agent/conversation_loop.py:2289`.

```python
while (api_call_count < agent.max_iterations
       and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
```

`_budget_grace_call`이 `or`로 붙어 있어서, 예산이 소진돼도 **마무리용 1턴 유예 호출**이 가능합니다.

### 이터레이션 구조

**1. 턴 중 유저 개입 흡수**

```python
_redirect_text = agent._drain_pending_redirect()
if _redirect_text:
    _apply_active_turn_redirect(agent, messages, _redirect_text)
    ...
    agent._persist_session(messages, conversation_history)
```

에이전트가 도구를 돌리는 중에 사용자가 새 메시지를 보내면, 다음 이터레이션 진입 시점에 흡수되어 `original_user_message`에 `User correction during the turn:` 형태로 합쳐집니다.

**2. API 호출**

`api_call_count += 1` → `agent._touch_activity(...)` → `step_callback` → `api_request_id = f"{turn_id}:api:{api_call_count}"`로 요청 발행.

**3. 응답 분기**

`finish_reason`에 따라 갈립니다.

- `tool_calls` → 도구 실행 (아래 6-1)
- `stop` → `_strip_think_blocks()` 후 최종 응답 후보
- `length` → 연속 이어붙이기(`length_continue_retries`, `truncated_response_parts`)

**4. 회복 경로들**

루프의 상당 부분이 프로바이더 이상 동작 복구입니다.

- **드롭된 tool-call** — Copilot의 claude-opus-4.8/sonnet-4.5가 `finish_reason="tool_calls"`인데 `tool_calls` 배열이 비어서 오는 케이스. 최대 3회 연속까지 재프롬프트하고, 성공한 도구 라운드 후 예산이 리셋됩니다. 재프롬프트 쌍은 `_dropped_toolcall_nudge` 플래그로 표시해 durable 트랜스크립트에 쓰이지 않게 합니다.
- **Codex ack 연속** — 중간 확인 응답을 최종 답으로 오인하지 않도록 `final_response = None`으로 되돌리고 계속.
- **압축 재시도** — `max_compression_attempts`(기본 3, `compression.max_attempts`)로 상한. 완료된 압축은 프로바이더 응답이 임계치 미만의 프롬프트를 보고해야만 카운터를 재무장합니다.
- **아웃터 루프 예외** — `_outer_error_count`가 `_MAX_OUTER_LOOP_ERRORS`를 넘으면 종료.

### 6-1. 도구 실행 — 세그먼트 플래너

단일 호출 지점(`conversation_loop.py:8158`)에서 `run_agent.py:9089`의 라우터로 갑니다.

```
_execute_tool_calls(assistant_message, messages, task_id, api_call_count)
  │
  ├─ len(tool_calls) <= 1 ────────────────→ _execute_tool_calls_sequential
  │
  └─ _plan_tool_batch_segments(...)
       ├─ 세그먼트 1개 & parallel ────────→ _execute_tool_calls_concurrent
       ├─ 세그먼트 1개 & sequential ──────→ _execute_tool_calls_sequential
       └─ 혼합 ───────────────────────────→ execute_tool_calls_segmented
```

세그먼트 플래너는 배치를 **병렬 안전 호출의 최대 연속 구간**(읽기 전용 도구, 겹치지 않는 파일 대상, 병렬 opt-in한 MCP 도구)으로 쪼개고, 그 사이에 순차 배리어(대화형·비안전·미인식 도구)를 둡니다. 동종 배치는 원래의 단일 경로를 유지하고, 혼합 배치만 발행 순서대로 세그먼트 단위 실행해서 안전한 부분집합은 병렬로 돌리면서 부수효과 순서는 보존합니다.

세 함수 모두 `agent/tool_executor.py`의 모듈 레벨 함수를 부르는 포워더이고, 최종적으로 `model_tools.handle_function_call()`에 도달합니다. `tool_executor.py`가 `_ra().handle_function_call`로 원래 모듈을 되짚는 이유는 테스트의 패치를 살리기 위해서입니다.

---

## Phase 7 — `finalize_turn()`

`agent/turn_finalizer.py:138`. 루프가 어떤 이유로 끝났든 여기를 통과합니다.

- `agent._save_trajectory(...)` — 파일 I/O + JSON 직렬화
- `agent._persist_session(messages, conversation_history)` — SQLite 쓰기
- 턴 종료 진단 로그 (`_turn_exit_reason`)
- 메모리 / 스킬 리뷰 넛지 (`_should_review_memory`, `_should_review_skills`)

각 정리 단계는 개별로 예외를 잡아 `_cleanup_errors`에 모읍니다. 예전에는 이 중 하나가 raise하면 결과 전체가 날아갔습니다.

반환 dict의 주요 키: `final_response`, `messages`, `completed`, `api_calls`, `error`, `partial`, `failed`, `turn_exit_reason`.

압축 타임아웃으로 끝난 턴은 반환 직전에 `error` / `partial` / `compression_exhausted`가 덮어씌워집니다.

---

## Phase 8 — `main()`으로 복귀

```python
print(f"✅ Completed: {result['completed']}")
print(f"📞 API Calls: {result['api_calls']}")
print(f"💬 Messages: {len(result['messages'])}")
```

`--save_sample`이 켜져 있으면 `agent._convert_to_trajectory_format(...)`으로 트라젝토리 형식(batch_runner와 동일)으로 변환해 `sample_<8자리>.json`에 저장합니다. 그리고 `👋 Agent execution completed!`로 종료.

---

## 전체 흐름 요약

```
python run_agent.py --query="..."
│
├─ [P1] 모듈 임포트
│    ├─ hermes_bootstrap (Windows UTF-8, guarded)
│    ├─ OpenAI/fire는 지연 (240ms 절약 + 부분설치 방어)
│    └─ model_tools → discover_builtin_tools() + discover_plugins()
│
├─ [P2] fire.Fire(main) — 시그니처가 곧 CLI 플래그
│    └─ --list_tools면 출력 후 조기 return
│
├─ [P3] AIAgent(...) — ~60개 파라미터 중 9개만 지정
│
└─ [P4] agent.run_conversation(query)          ← hermes 경로와 합류
     │
     ├─ 포워더 (run_agent.py:9272)
     │    ├─ 백그라운드 리뷰 선점
     │    ├─ 리뷰 큐 라이브니스 마킹
     │    ├─ durable turn lease (state.db 직렬화, fail-closed)
     │    └─ relay/accounting/portal 컨텍스트
     │
     ├─ [P5] build_turn_context()  (turn_context.py:572)
     │    ├─ 시스템 프롬프트 restore-or-build (캐시 보존)
     │    ├─ preflight 압축 → 타임아웃 시 타입 있는 조기 반환
     │    ├─ pre_llm_call 훅 + 외부 메모리 prefetch
     │    └─ api_mode=codex_app_server면 여기서 완전 우회
     │
     ├─ [P6] while (예산 남음) or (유예 호출)   (conversation_loop.py:2289)
     │    ├─ 턴 중 유저 개입 흡수 (_drain_pending_redirect)
     │    ├─ API 호출
     │    ├─ finish_reason 분기: tool_calls / stop / length
     │    ├─ 도구 실행 → 세그먼트 플래너 → tool_executor
     │    │                              → handle_function_call
     │    └─ 회복: 드롭된 tool-call, Codex ack, 압축 재시도
     │
     └─ [P7] finalize_turn()  (turn_finalizer.py:138)
          ├─ _save_trajectory
          ├─ _persist_session
          └─ 메모리/스킬 리뷰 넛지
```

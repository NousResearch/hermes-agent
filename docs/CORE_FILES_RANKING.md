# 핵심 비즈니스 로직 파일 5개 — 랭킹 근거

> 분석 기준일: 2026-09-04 · 대상 커밋: `63279301bc` (main)
> 랭킹 기준은 **파일 크기가 아니라 레버리지** — 여기 한 줄을 바꿨을 때 시스템의 몇 퍼센트가 따라 바뀌는가.

| 순위 | 파일 | LOC | 역할 |
| --- | --- | --- | --- |
| **1** | `agent/conversation_loop.py` | 9,339 | 한 번의 유저 턴을 끝까지 굴리는 실제 엔진 — 모델 호출, 툴 디스패치, 재시도/페일오버, 압축, 턴 후 훅, 메모리·스킬 리뷰 넛지 |
| **2** | `run_agent.py` | 10,175 | `AIAgent` 클래스 본체 — 에이전트의 모든 상태(~60개 생성자 파라미터: 크리덴셜, 라우팅, 예산, 세션, 콜백)를 소유하는 객체 |
| **3** | `model_tools.py` | 1,707 | `handle_function_call()` — 모든 툴 호출이 반드시 통과하는 단일 관문(플러그인 훅, 미들웨어, 관측성, 에러 분류) |
| **4** | `hermes_state.py` | 17,370 | `SessionDB` — WAL + FTS5 SQLite 세션 스토어, 대화가 프로세스 밖에서 살아남는 유일한 지점 |
| **5** | `agent/context_compressor.py` | 9,246 | 컨텍스트 압축 — 대화가 컨텍스트 윈도우를 넘어서도 계속되게 만드는 로직 |

---

## 1위와 2위를 가른 기준

두 파일은 **"상태 소유자 vs 행위 소유자"**로 쪼개져 있습니다. `conversation_loop.py`의 함수들은 전부 `agent: Any`를 첫 인자로 받아 속성 조회로 부모 `AIAgent`의 상태에 접근합니다:

```python
def _maybe_inject_run_budget_wrapup(agent: Any, messages: List[Dict[str, Any]]) -> bool:
def _should_rearm_compression_budget(agent, ...)
def _restore_or_build_system_prompt(agent, system_message, conversation_history):
```

에이전트가 **무엇을 하는가**는 1위에, **무엇을 가지고 있는가**는 2위에 있습니다. 행위가 더 중요하다고 봐서 1위로 뒀습니다.

같은 추출 패턴이 `agent/tool_executor.py`(2,940 LOC — 순차/동시 툴 디스패치)에도 적용돼 있습니다. `run_agent.py`가 `_ra()` 헬퍼로 원래 모듈을 되짚는 이유는 `run_agent._set_interrupt`, `handle_function_call`, `OpenAI` 같은 심볼을 패치하는 기존 테스트를 깨지 않기 위해서입니다.

---

## 3위가 1,700줄인데 3위인 이유

크기가 아니라 **레버리지**입니다. 모든 툴 호출의 단일 병목이라 여기 한 줄을 바꾸면 40개 툴 전부의 동작이 바뀝니다. `handle_function_call()`의 시그니처가 그 사실을 드러냅니다 — `skip_pre_tool_call_hook`, `skip_tool_request_middleware`, `skip_tool_execution_middleware` 같은 파라미터는 이 함수가 훅/미들웨어 파이프라인의 주인이라는 뜻입니다.

---

## 4위: 왜 17k LOC짜리가 4위인가

절대량은 1등이지만 상당 부분이 SQLite 운영 방어 로직(재시도, WAL 복구, 예외 계층 10종 — `DeletedWalGenerationError`, `SessionTurnLeaseLostError` 등)입니다. 다만 파일 상단 docstring이 밝히는 설계 결정들은 진짜 도메인 로직입니다:

- 게이트웨이 멀티플랫폼 동시 접근을 위한 WAL 모드(다중 리더 + 단일 라이터)
- **압축이 트리거하는 세션 분할** — `parent_session_id` 체인
- 배치 러너와 RL 트라젝토리는 여기 저장하지 **않음**(별도 시스템)

---

## 5위: `toolsets.py`가 아니라 `context_compressor.py`인 이유

`toolsets.py`(1,062 LOC)도 후보였습니다 — 매 API 호출에 어떤 툴 스키마가 실리는지를 결정하니 "좁은 허리" 원칙과 비용을 직접 지배하죠. 하지만 그건 본질적으로 선언적 딕셔너리입니다.

`context_compressor.py`를 고른 건 **이 프로젝트가 신성불가침이라 선언한 프롬프트 캐싱 규칙의 유일한 예외**이기 때문입니다. AGENTS.md가 "과거 컨텍스트를 변형하는 건 하지 않는다 — 유일한 예외가 컨텍스트 압축"이라고 못박은 그 예외의 구현체이고, 장수 대화라는 제품 핵심을 성립시키는 알고리즘이 여기 있습니다.

---

**차점자:** `toolsets.py`(1,062), `agent/tool_executor.py`(2,940), `tools/registry.py`(1,372), `agent/system_prompt.py`(1,172)

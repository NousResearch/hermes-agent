# Rust ROI Sequential Plan

## 목적

이 작업의 목적은 Python을 부분 또는 전부 대체하는 Rust 경로가 실제 운영 ROI를 낼 정도로 월등한 성능 개선을 만드는지 검증하고, 그 개선이 없으면 빠르게 중단하는 것입니다.

핵심 원칙은 아래와 같습니다.

- 소폭 개선은 실패로 간주한다.
- 각 단계는 테스트를 통과해야 다음 단계로 진행한다.
- 최종 단계에서는 대상 모듈 단위가 아니라 전체 플로우에서 기존 Python 경로를 완전히 대체할 수 있어야 한다.
- 성능 개선이 충분하지 않으면 Python 유지가 정답이다.

## 비목표

- Rust 도입 자체를 목표로 하지 않는다.
- Python과 Rust를 영구적으로 이중 유지하는 상태를 목표로 하지 않는다.
- 벤치마크 숫자만 좋아지는 최적화를 목표로 하지 않는다.

## 최종 성공 기준

최종 채택은 아래 조건을 모두 만족해야 합니다.

1. `ResponseStore` 또는 `Compression` 치환 중 하나 이상에서 ROI 급 개선이 나온다.
2. 전체 플로우 테스트에서 Rust 경로가 기존 Python 경로를 기능적으로 대체한다.
3. 장애, 재시작, 장시간 실행 테스트에서 운영 안정성을 유지한다.
4. 성능 개선이 아래 컷라인 중 하나를 만족한다.

- `p95` 또는 `p99` 지연시간이 2배 이상 개선
- CPU 또는 RSS 사용량이 40% 이상 절감
- 동시 처리 한계가 2배 이상 증가

위 조건을 만족하지 못하면 확대 포팅을 중단하고 Python 유지로 결론냅니다.

## 단계 개요

### 1단계. 설계서 및 스캐폴드 고정

산출물:

- 본 설계 문서
- Rust 워크스페이스 스캐폴드
- Python-Rust sidecar 경계용 최소 프로토콜 타입
- health 응답이 가능한 최소 sidecar 바이너리

필수 테스트:

- `cargo test --manifest-path rust/Cargo.toml`
- `cargo run --manifest-path rust/Cargo.toml -p hermes-sidecar -- --health`

통과 조건:

- 프로토콜 타입이 직렬화/역직렬화 테스트를 통과한다.
- sidecar health 응답이 JSON으로 출력된다.
- 이후 단계에서 재사용할 RPC 경계와 ROI 컷라인이 문서로 고정된다.

### 2단계. ResponseStore 치환

대상:

- `gateway/platforms/api_server.py`의 `ResponseStore`

필수 테스트:

- 단위 테스트: put/get/delete/conversation mapping/LRU eviction
- 동등성 테스트: 기존 Python store와 동일 입력 동일 출력
- 동시성 테스트: 다중 reader/writer 충돌 없음
- 지속성 테스트: 재시작 후 복구
- 전체 플로우 테스트: `/v1/responses`, `previous_response_id`, `GET`, `DELETE`

통과 조건:

- 기존 Python `ResponseStore` API와 결과가 완전히 동일하다.
- 전체 Responses API 플로우에서 Python 구현을 끄고도 동일하게 동작한다.
- 아래 중 하나를 만족한다.
  - `p95` read/write latency 2배 이상 개선
  - CPU 또는 RSS 40% 이상 절감
  - 고동시 요청 수용량 2배 이상 증가

중단 조건:

- 기능 동등성이 깨진다.
- 장애 복구가 불안정하다.
- ROI 개선이 컷라인에 못 미친다.

### 3단계. Compression 치환

대상:

- `trajectory_compressor.py`
- `agent/context_compressor.py`

필수 테스트:

- 단위 테스트: 구조 보존, 보호 구간, 요약 삽입 규칙
- 동등성 테스트: 압축 후 메시지 시퀀스 무결성
- 품질 테스트: tool call/result 쌍 보존, 필수 메시지 보존
- 성능 테스트: 대화 길이별 처리시간과 메모리 사용량
- 전체 플로우 테스트: 긴 대화에서 자동 압축 후 에이전트가 정상 진행

통과 조건:

- 구조적 무결성 오류가 0건이다.
- 전체 플로우에서 Rust 경로가 Python 압축 경로를 끄고도 대체된다.
- 에이전트 성공률 저하가 2%p 이하이다.
- 아래 중 하나를 만족한다.
  - `p95` 압축 시간 2배 이상 개선
  - CPU 또는 RSS 40% 이상 절감
  - 장문 대화 처리량 2배 이상 증가

중단 조건:

- 압축 후 대화 무결성이 깨진다.
- 에이전트 결과 품질이 의미 있게 떨어진다.
- ROI 개선이 컷라인에 못 미친다.

### 4단계. 전체 플로우 대체 검증

대상:

- `run_agent.py`
- `gateway/run.py`
- `gateway/platforms/api_server.py`

필수 테스트:

- end-to-end 회귀 테스트
- Rust sidecar only 모드 테스트
- 장애 주입 테스트: sidecar 재시작, 타임아웃, 잘못된 응답
- soak 테스트: 30분 이상 연속 부하
- 운영 시나리오 테스트: 저장소 복원 + 압축 + 응답 체이닝

통과 조건:

- 전체 플로우에서 기존 Python 모듈을 비활성화해도 동일 기능이 유지된다.
- 치환 대상 구간에서 fallback 없이 Rust가 주 경로가 된다.
- soak 동안 메모리 증가와 에러율이 허용 범위 내다.
- 최종 ROI 컷라인을 만족한다.

중단 조건:

- 전체 플로우에서 기능 동등성이 확보되지 않는다.
- 운영형 부하에서 안정성이 떨어진다.
- ROI가 최종 컷라인을 넘지 못한다.

## DO Loop 의사결정

Rust 검증은 길게 끌지 않고 아래 순서로 진행합니다.

1. `ResponseStore` 치환
2. `Compression` 치환

위 두 루프 안에서 ROI 급 개선이 확인되지 않으면 전체 계획을 중단하고 Python 유지로 결론냅니다.

## 테스트 계층

### 단위 테스트

- Rust 내부 타입, 스토어 연산, 압축 로직의 순수 함수 검증

### 계약 테스트

- Python이 기대하는 입력/출력 스키마와 Rust sidecar 응답 형식 검증

### 동등성 테스트

- 동일 입력에 대해 Python 구현과 Rust 구현의 결과 비교

### 부하 테스트

- 동시 요청 수 증가에 따른 지연시간, CPU, RSS, 오류율 측정

### Soak 테스트

- 장기 실행에서 메모리 증가량, 오류 누적, 복구 가능성 측정

### 전체 플로우 테스트

- Responses API
- 자동 압축
- 이전 응답 체이닝
- 재시작 후 복구
- Python fallback 없이 Rust만으로 동작하는지 검증

## Python-Rust 경계 규칙

- Python은 오케스트레이터로 유지한다.
- Rust는 sidecar 프로세스로 분리한다.
- 1차 프로토콜은 JSON over stdio 또는 Unix domain socket 기반 envelope를 사용한다.
- 모든 요청은 `version`, `id`, `method`, `params`를 가진다.
- 모든 응답은 `version`, `id`, `ok`와 `result` 또는 `error`를 가진다.
- 최종 대체 검증 단계 전까지는 Python fallback을 유지하되, 단계 테스트에서는 반드시 fallback off 모드를 따로 검증한다.

## 마일스톤별 머지 원칙

- 각 단계는 `중간머지`에서 분기한다.
- 각 단계는 해당 범위만 수정한다.
- 테스트 실패 시 같은 단계 브랜치에서 수정 후 재검증한다.
- 통과 전에는 다음 단계로 넘어가지 않는다.


# Hermes Python-only 성능 최적화 개발 완료보고서

## 1. 문서 목적

이 문서는 `NousResearch/hermes-agent`에 대해 수행한 Python-only 성능 최적화 작업의 최종 결과를 정리한 완료보고서입니다.

정리 범위는 아래를 포함합니다.

- 처음 시도했던 Rust 방향과 중단 이유
- 실제 채택된 Python-only 최적화 범위
- 구현 상세
- 테스트 및 검증 절차
- 벤치마크 수치와 해석
- 현재 PR 상태
- 남은 리스크와 후속 관찰 포인트

## 2. 작업 배경

초기 문제의식은 단순했습니다.

- `Responses API` hot path에서 저장/조회 경로가 상대적으로 무겁다
- 긴 대화 체인과 누적된 응답 기록이 있을수록 부담이 커질 수 있다
- gateway에서 같은 media URL을 동시에 여러 번 처리할 때 중복 fetch가 발생할 수 있다

원래는 Rust 포팅 가능성도 검토했습니다. 그러나 실제 실험 결과, 부분적인 Rust sidecar 치환은 ROI가 나오지 않았습니다.

핵심 이유는 다음과 같았습니다.

- Python 프로세스와 Rust sidecar 사이의 경계 비용이 컸다
- JSON serialize/deserialize와 subprocess/pipe 왕복 비용이 누적되었다
- 작은 단위의 작업을 sidecar로 넘기는 구조에서는 계산 이득보다 RPC 비용이 더 컸다

따라서 방향을 아래처럼 바꿨습니다.

- 언어를 바꾸는 대신 Python 경로에서 실제 병목 데이터를 더 가볍게 만든다
- 저장 구조를 바꾼다
- 중복 작업을 줄인다
- 성능이 안 나오는 실험은 최종 브랜치에 남기지 않는다

## 3. 최종 목표

최종 목표는 "코드가 더 멋져 보이는 구조"가 아니라 아래였습니다.

1. 실제 `Responses API` 요청 경로를 유의미하게 빠르게 만들 것
2. 실패율이나 의미론을 망가뜨리지 않을 것
3. 운영 복잡도를 크게 늘리지 않을 것
4. 재현 가능한 방법으로 수치를 입증할 것

## 4. 최종 채택 범위

최종적으로 채택된 변경은 아래 세 가지입니다.

### 4.1 ResponseStore 저장 구조 최적화

기존에는 응답 저장 시 대화 history를 사실상 전체 덩어리로 더 무겁게 다루는 경향이 있었습니다.

이를 아래 구조로 변경했습니다.

- 기준점 역할을 하는 snapshot history node 유지
- 이후 응답은 delta history node로 연결
- `previous_response_id` 체인은 유지
- 부모 응답이 삭제되어도 자식 체인 복원이 깨지지 않도록 history node를 독립적으로 유지

결과적으로 기대한 효과는 아래였습니다.

- 응답 저장 시 전체 history를 매번 크게 다루는 부담 감소
- 응답 조회 시 reconstruction 비용 제어
- 긴 체인에서 저장/조회 비용의 증가 폭 완화

주요 수정 파일:

- `gateway/platforms/api_server.py`
- `tests/gateway/test_api_server.py`

### 4.2 Gateway media download dedupe

같은 media URL에 대해 동시에 여러 요청이 들어오면, 구현에 따라 같은 파일을 여러 번 다시 다운로드하는 낭비가 생길 수 있습니다.

이를 줄이기 위해 아래를 추가했습니다.

- same-key singleflight
- short-lived result cache
- cache key에 cache dir까지 포함해 환경 간 오염 방지

기대한 효과:

- 동일 media에 대한 중복 upstream fetch 제거
- concurrency 상황에서 외부 I/O 낭비 감소
- gateway hot path 안정성 개선

주요 수정 파일:

- `gateway/platforms/base.py`
- `tests/gateway/test_media_download_retry.py`

### 4.3 Compression 경로는 보수적으로 정리

중간 단계에서 `ContextCompressor`에 일반화된 micro-cache를 넣는 실험을 했습니다.

그러나 최종 microbenchmark에서 이 경로는 이득보다 회귀가 더 컸습니다. 그래서 최종 브랜치에서는 해당 실험을 제거했습니다.

즉 최종 브랜치는 아래 원칙으로 정리했습니다.

- 실제로 이득이 측정된 변경은 유지
- 미세하게라도 회귀가 확인된 실험성 최적화는 제거

관련 파일:

- `agent/context_compressor.py`
- `tests/agent/test_context_compressor.py`

## 5. 버린 것

이번 작업에서 중요한 점은 "추가한 것"만이 아니라 "버린 것"도 명확하다는 점입니다.

### 5.1 Rust sidecar 방향 중단

중단 이유:

- sidecar 구조에서는 IPC와 JSON 경계 비용이 너무 컸다
- `ResponseStore`와 `ContextCompressor` 같은 작은 작업 단위에서는 Python 인프로세스 경로보다 느렸다
- 운영 복잡도와 유지 비용까지 고려하면 투자 대비 효과가 부족했다

정리:

- Rust 코드는 최종 브랜치에서 제거
- Python-only 경로만 남김

### 5.2 Compression micro-cache 실험 제거

중단 이유:

- microbenchmark 기준 미세 최적화가 실제로는 회귀를 만들었다
- serving-path 이득과 직접 연결되지 않았다
- 유지할 이유가 없었다

정리:

- 해당 캐시 실험은 롤백
- 최종 브랜치는 보수적으로 유지

## 6. 실사용 기준으로 어디가 빨라지는가

이번 최적화는 사용자 입장에서 아래 상황에서 의미가 큽니다.

1. 이전 대화를 이어서 계속 요청할 때
- 이전 응답과 대화 history를 이어 붙여 처리하는 경로가 더 가벼워집니다.

2. 저장된 응답을 다시 조회할 때
- `GET /v1/responses/{id}` 경로가 크게 개선되었습니다.

3. 응답 기록이 누적된 세션일 때
- history 체인이 길어질수록 저장/조회 경로 최적화의 효과를 보기 좋습니다.

4. 같은 media URL을 여러 경로에서 동시에 처리할 때
- 중복 다운로드를 줄여 외부 fetch 낭비를 낮춥니다.

즉, "한 번 요청하고 끝나는 아주 짧은 사용 패턴"보다, 대화를 이어가고 저장된 응답을 다시 불러오는 실제 에이전트 사용 패턴에서 더 의미가 있습니다.

## 7. 벤치마크 설계

성능 수치에서 가장 중요하게 본 것은 숫자 자체보다 측정 신뢰성이었습니다.

로컬 성능 측정은 쉽게 오염됩니다. 특히 이 저장소처럼 로컬 상태(`~/.hermes/response_store.db`)를 사용하는 경우, 브랜치 간 상태 공유만으로도 결과가 왜곡될 수 있습니다.

그래서 최종 수치는 아래 조건에서 다시 측정했습니다.

### 7.1 측정 환경

- 동일 머신
- 동일 benchmark server harness
- 동일 요청 payload
- 동일 동시성
- 동일 ApacheBench 조건

### 7.2 오염 방지 조치

- `main`용 별도 `git worktree`
- 브랜치별 별도 `HOME`
- 두 브랜치가 `~/.hermes/response_store.db`를 공유하지 않도록 격리

### 7.3 API benchmark 조건

- `ab -n 1000 -c 50`
- `POST /v1/responses`
- `GET /v1/responses/{id}`

### 7.4 수집 지표

- latency
- throughput
- failed count
- non-2xx count
- RSS

## 8. 최종 벤치마크 결과

기준 비교는 `main` vs 최종 PR 브랜치입니다.

### 8.1 `POST /v1/responses`

- `main`: `98.650 ms/request`, `506.84 req/s`
- 최종 브랜치: `67.667 ms/request`, `738.91 req/s`
- 개선폭: 약 `1.46x`

해석:

- 새 응답 생성 경로가 유의미하게 빨라졌습니다.
- 단순 오차 범위 수준이 아니라, 실제 serving path에서 의미 있는 차이입니다.

### 8.2 `GET /v1/responses/{id}`

- `main`: `56.838 ms/request`, `879.69 req/s`
- 최종 브랜치: `29.723 ms/request`, `1682.21 req/s`
- 개선폭: 약 `1.91x`

해석:

- 저장된 응답 조회 경로는 거의 2배에 가까운 개선이 나왔습니다.
- 이번 작업에서 가장 눈에 띄는 성과입니다.

### 8.3 실패율

- `POST`: `failed=0`, `non-2xx=0`
- `GET`: `failed=0`, `non-2xx=0`

해석:

- 단순히 빨라졌을 뿐 아니라, 벤치 중 실패율이 같이 나빠지지 않았습니다.
- 성능 개선과 의미론 안정성을 함께 유지했다는 점에서 중요합니다.

### 8.4 메모리

- `main`: `37384 KB`
- 최종 브랜치: `36964 KB`
- 변화: 약 `1.1%` 감소

해석:

- 메모리 개선은 크지 않습니다.
- 이번 작업의 핵심 이득은 메모리보다 API hot path latency/throughput에 있습니다.

### 8.5 Media dedupe synthetic benchmark

동일 URL에 대한 `25`개 동시 요청 기준:

- `main`: `network_calls=25`, `unique_paths=25`
- 최종 브랜치: `network_calls=1`, `unique_paths=1`
- 중복 upstream fetch 감소: `96%`

해석:

- mock wall-clock 하나만 보면 더 빠르다고 단정하긴 어렵습니다.
- 그러나 운영 관점에서는 redundant upstream fetch collapse 자체가 유의미합니다.
- 외부 I/O 낭비를 크게 줄인다는 점에서 가치가 있습니다.

### 8.6 Compression microbenchmark

`100`회 반복 기준:

- `main`: median `0.555 ms`, p95 `4.791 ms`
- 최종 브랜치: median `0.656 ms`, p95 `5.130 ms`

해석:

- compression 경로는 소폭 회귀가 있습니다.
- 그래서 이 경로를 더 억지로 밀지 않았고, micro-cache 실험을 제거한 상태에서 보수적으로 정리했습니다.
- 본 PR의 핵심 가치는 compression이 아니라 `Responses API` hot path 개선입니다.

## 9. 기능 검증 및 테스트

성능 수치만으로는 충분하지 않기 때문에, 의미론 회귀와 기능 회귀를 막기 위한 테스트도 함께 수행했습니다.

### 9.1 ResponseStore / semantic behavior

실행 명령:

```bash
python -m py_compile gateway/platforms/api_server.py tests/gateway/test_api_server.py
pytest -o addopts='' tests/gateway/test_api_server.py::TestResponseStore tests/gateway/test_api_server.py::TestResponsesEndpointSemantic -q
```

결과:

- `11 passed`

검증한 핵심 동작:

- `previous_response_id` chaining
- 이전 응답으로부터 instruction inheritance
- 부모 응답 삭제 후에도 child-chain reconstruction 유지

### 9.2 Context compressor

실행 명령:

```bash
python -m py_compile agent/context_compressor.py tests/agent/test_context_compressor.py
pytest -o addopts='' tests/agent/test_context_compressor.py::TestShouldCompressPreflight tests/agent/test_context_compressor.py::TestCompress tests/agent/test_context_compressor.py::TestGetStatus -q
```

결과:

- `8 passed`

설명:

- compression 경로는 최적화 실험을 과감히 줄이고, 기존 동작의 안정성을 유지하는 방향으로 정리했습니다.

### 9.3 Media retry/dedupe + semantic response flow

실행 명령:

```bash
python -m py_compile gateway/platforms/base.py tests/gateway/test_media_download_retry.py
pytest -o addopts='' tests/gateway/test_media_download_retry.py::TestCacheImageFromUrl tests/gateway/test_media_download_retry.py::TestCacheAudioFromUrl tests/gateway/test_api_server.py::TestResponsesEndpointSemantic tests/test_anthropic_error_handling.py -q
```

결과:

- `24 passed`

설명:

- media dedupe와 semantic response flow를 동시에 검증했습니다.

## 10. 최종 해석

이번 결과는 아래처럼 해석하는 것이 가장 정확합니다.

### 10.1 성공한 부분

- `Responses API` hot path는 실제로 의미 있게 빨라졌다
- `GET /v1/responses/{id}`는 거의 `2x` 수준까지 개선됐다
- `POST /v1/responses`도 충분히 의미 있는 개선이 나왔다
- 실패율 회귀 없이 측정되었다
- 중복 media fetch를 크게 줄였다

### 10.2 의도적으로 포기한 부분

- Rust sidecar 방향
- compression micro-cache 방향

### 10.3 결론

이 작업은 "모든 microbenchmark를 개선한 만능 최적화"가 아닙니다.

대신 아래에 가깝습니다.

- 실제 서비스 경로에서 의미 있는 성능 개선을 낸 변경만 남김
- ROI가 낮거나 회귀가 있던 방향은 제거
- 보수적이지만 설득 가능한 최적화 세트로 정리

즉, 코드 양을 많이 늘리거나 화려한 재작성으로 승부한 것이 아니라, 실사용 경로에서 검증된 이득만 남기는 방식으로 정리한 PR입니다.

## 11. PR 현황

현재 PR은 아래에 올라가 있습니다.

- PR: `https://github.com/NousResearch/hermes-agent/pull/4215`

제목:

- `Optimize Responses API hot path without Rust sidecar`

현재 상태 요약:

- PR은 open 상태
- 충돌은 없음
- 최종 머지는 아직 아님
- 리뷰 및 maintainer 판단이 남아 있음

## 12. 남은 리스크와 관찰 포인트

### 12.1 Compression 경로

- 현재 PR의 핵심 성과는 아니지만, microbenchmark 기준 소폭 회귀가 남아 있습니다.
- 다만 실제 가치는 `Responses API` hot path 개선에 있으므로, 이번 PR에서는 확장 최적화보다 범위 통제가 더 중요했습니다.

### 12.2 운영 환경에서의 media dedupe 효과

- synthetic benchmark에서는 wall-clock 이득이 직접적으로 크게 드러나지 않았습니다.
- 그러나 실제 운영 환경에서는 upstream fetch collapse 효과가 더 중요할 수 있습니다.
- 머지 후 실제 트래픽 환경에서 fetch 절감 효과를 관찰하는 것이 좋습니다.

### 12.3 실제 사용자 체감

- 사용자는 주로 아래 상황에서 체감할 가능성이 큽니다.
- 긴 세션에서 이어서 대화할 때
- 저장된 응답을 다시 조회할 때
- media URL이 반복적으로 등장할 때

## 13. 최종 요약

이번 작업의 핵심은 아래 한 문장으로 요약할 수 있습니다.

"Rust로 크게 갈아엎는 대신, Python 경로에서 실제 병목인 저장/조회와 중복 fetch를 줄여서, `Responses API` hot path를 측정 가능한 수준으로 개선했다."

요약 수치:

- `POST /v1/responses`: 약 `1.46x` 개선
- `GET /v1/responses/{id}`: 약 `1.91x` 개선
- `failed=0`, `non-2xx=0`
- media duplicate upstream fetch: `96%` 감소

이번 PR은 대규모 인기 오픈소스 프로젝트에서 "실서비스 경로 기준으로 재현 가능한 성능 개선"을 입증했다는 점에서 의미가 있습니다.

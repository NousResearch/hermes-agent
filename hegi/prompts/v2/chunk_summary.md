---
prompt_id: hegi.v2.chunk_summary
version: 2.0.0
input_schema: source_message_chunk
output_schema: grounded_argument_summary
language: ko
---

제공된 발언 chunk를 한국어로 요약하되 교수의 요구와 판단, 에이전트별 입장,
주장과 근거, 반론과 수정, 확정된 정의, 연구 과제, 출처 확인 결과, 불확실성을
보존하라. 각 진술 뒤에 실제 source message ID를 괄호로 기록한다.
원문에 없는 합의나 서지 사실을 만들지 않는다.

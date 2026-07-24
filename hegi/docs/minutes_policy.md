# HEGI 회의록 작성·출력 정책

이 정책은 프롬프트 권고가 아니라 collector, analyzer, renderer, Memory gate와 최종
quality gate에서 강제된다.

1. 모든 `user` 메시지는 교차 DB 중복 제거 후 화자를 `교수`로 정규화한다.
2. dict/list를 문자열로 변환한 Python repr는 구조화 필드에 수용하거나 Markdown에
   출력하지 않는다.
3. 모든 episode를 `research_meeting`, `operational_incident`, `mixed`, `other` 중
   하나로 분류한다.
4. `operational_incident`는 연구회의가 아닌 장애 개요, 복구 과정, 해결 상태,
   행동 로그 중심의 별도 템플릿을 사용한다.
5. 명령 실행·파일 수정·재시작·테스트는 `agent_activity_log`, 개념적 판단과 연구
   제안은 `agent_positions`에 기록한다.
6. Memory Forest 직접 쓰기, 쓰기 활성화, 자동 승인 또는 자동 Commit은 Action Item
   으로 생성하지 않는다.
7. Memory Evaluation은 실행한 검색, 검색 결과 수, 중복 후보 ID·제목·관계,
   신규성 근거와 최종 판정 이유를 출력한다.
8. 연구적 지속성이 없거나 운영 장애인 episode는 `no_memory`로 판정하며, 교수 승인
   메시지가 있어도 STM Draft를 만들지 않는다.
9. 오래된 발언과 현재 상태가 충돌하면 `temporal_conflicts`에 이전 상태, 현재 상태,
   해결 여부와 근거 message ID를 기록한다.
10. Archive와 Telegram 출력 전 quality gate에서 raw Python repr, 허용되지 않은
    화자명, Memory Forest 직접 쓰기 안전정책 위반 문구를 발견하면 해당 episode의
    출력과 Action Item 저장을 차단하고 dead letter에 기록한다.

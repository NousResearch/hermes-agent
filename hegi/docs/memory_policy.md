# Memory Safety Policy

HEGI 자동화 계층은 Memory Forest read와 curator draft만 사용한다. 직접 STM write,
approval, commit 도구를 등록하거나 호출하지 않는다.

Draft 생성 조건은 모두 충족되어야 한다.

1. 설정된 교수 Telegram user ID가 승인한다.
2. 승인 문구가 지원하는 네 명령 중 하나다.
3. platform message ID가 이전에 처리되지 않았다.
4. 대상 meeting과 저장된 회의록이 존재한다.
5. Draft 직전에 Memory Forest를 다시 검색한다.
6. 승인 명령이 기억·초안·병합 중 하나다. “기억하지 마”는 Draft를 차단한다.

생성 결과는 pending draft다. 이후 approve/commit은 HEGI 범위 밖의 별도 인간 검토
절차다.

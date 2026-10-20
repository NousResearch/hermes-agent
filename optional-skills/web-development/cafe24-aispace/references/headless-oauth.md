# Headless OAuth for AI SPACE MCP (Hermes / server environments)

Validated procedure (field-tested on Hermes Agent, July 2026, two runs). Korean original follows — the steps are framework-agnostic.

Server URL: `https://aih-proxy.cafe24.com/mcp` (streamable-http, OAuth 2.0 DCR + PKCE; access token ~15 min, refresh required).

---

### 에이전트 프레임워크(OpenClaw·Hermes 등 헤드리스 서버)에서 쓸 때

브라우저가 없는 서버 환경은 일반 도구와 인증 동작이 다르다. 최초 설치는 아래 순서가 정석이다.

**최초 설치 4단계**

1. **등록**: 원격 MCP를 `transport: streamable-http` + `auth: oauth`로 등록한다 (`https://aih-proxy.cafe24.com/mcp`).
2. **최초 인증 — URL 붙여넣기 방식(기본)**: 인증 URL을 사용자에게 제시한다 → 사용자가 브라우저에서 열어 카페24 로그인·승인한다 → **승인 후 이동된 페이지가 연결 오류로 떠도 정상이다**(인증 코드는 주소창 URL에 있다) → 사용자에게 "주소창의 URL 전체를 복사해서 붙여넣어 달라"고 요청한다 → 받은 URL로 토큰 교환을 완료한다.
   - **토큰 교환에 성공하면 출력하지 말고 즉시 프레임워크 토큰 경로에 파일로 저장한다.** 저장을 확인한 뒤에만 성공을 보고할 것 — 출력만 하면 토큰이 증발해 사용자에게 브라우저 재인증을 반복시키게 된다.
   - 예외: 인증 URL을 여는 단계 자체가 거부되면(redirect_uri 불일치) 설정에서 redirect URL을 고정값으로 오버라이드하거나, 사전 등록 커넥터(고정 redirect + 서버측 토큰 주입)를 쓴다.
3. **갱신 설정**: access 토큰 수명은 약 15분이다. refresh 토큰 자동 갱신을 반드시 켠다 — 없으면 15분 뒤부터 인증이 계속 끊긴다. "예전엔 됐는데 지금 안 된다"의 최다 원인. 갱신 잡 설계 규칙:
   - **에이전트 잡(대화 루프 안 크론)으로 걸지 말 것.** 프레임워크의 에이전트 크론은 매 실행 결과를 채팅(Telegram 등)으로 배달해 사용자를 괴롭힌다. **OS crontab이나 systemd 타이머에서 독립 스크립트로** 실행한다.
   - **성공 시 무출력** — 결과는 로그 파일에만 남긴다. 예: `*/10 * * * * /path/refresh-aispace.sh >> ~/aispace-refresh.log 2>&1`
   - 실패 알림도 채팅으로 반복 발송하지 말 것 — 로그에 남기고, 다음 도구 호출이 401을 만나면 아래 "401 처리 순서"가 잡는다. 알리더라도 연속 실패가 지속될 때 1회만.
   - 스크립트 내용: 토큰 파일에서 refresh_token을 읽어 토큰 엔드포인트(디스커버리로 확인)에 refresh grant 요청 → 응답 토큰을 **같은 파일에 원자적으로 덮어쓰기**.
4. **스모크 테스트**: `list_my_projects`를 1회 호출해 성공을 확인하고 시작한다.

**운영 중 401/인증 에러 처리 순서**: ① "Auth: none"으로 표시되면 토큰 문제가 아니라 **설정 문제**다(아래 함정 목록) → ② refresh 토큰으로 갱신 시도 → ③ 실패하면 2단계(URL 붙여넣기 재인증)를 다시 안내 → ④ **같은 인증 에러로 2회 연속 실패하면 재시도를 멈추고 보고한다.** 인증 에러를 배포 실패로 오진해 코드를 고치지 말 것.

**알려진 함정 (Hermes 실측 2026-07, 2회 검증 — 버전에 따라 다를 수 있음)**

- headless에서 `mcp add`가 인증 URL조차 출력하지 않고 실패할 수 있다(OAuthNonInteractiveError) → OAuth 플로우를 수동 수행: `.well-known/oauth-authorization-server` 디스커버리 → 동적 클라이언트 등록 → PKCE → 인증 URL을 사용자에게 제시 → 콜백 URL 붙여넣기로 토큰 교환.
- **`mcp add --auth oauth`가 실패하면서 config를 `auth: none`으로 저장해 버릴 수 있다** → add 실행 후 config의 auth 필드를 반드시 확인하고, 틀려 있으면 config를 직접 수정한다(예: `config set mcp_servers.<서버명>.auth oauth`). 이게 "토큰은 멀쩡한데 401 + Auth: none"의 진짜 원인이다.
- 수동 OAuth의 단계별 산출물(클라이언트 등록 정보·PKCE verifier·토큰)은 **생성 즉시 파일로 저장**한다. 메모리/출력에만 두면 중간 실패 시 처음부터(사용자 재인증 포함) 다시 해야 한다.
- 동적 등록 전에 **실제 가용 포트를 먼저 할당**하고 그 포트로 redirect_uri를 등록할 것. 포트 0(`127.0.0.1:0/callback`)으로 등록하면 검증 400으로 실패한다.
- config에는 `auth: oauth` 필드를 **명시**해야 OAuth 프로바이더가 생성된다. `oauth:` 자격증명 블록만 넣으면 "Auth: none"(토큰 미첨부) 상태로 401이 난다.
- 수동 발급한 토큰은 프레임워크가 읽는 **정확한 경로와 JSON 키 이름**으로 저장할 것(예: `mcp-tokens/<서버명>.json`). 위치나 스키마가 다르면 조용히 무시된다.
- config 변경은 **새 세션(또는 MCP 리로드)에서만 반영**된다. 수정 후 기존 세션에서 테스트하며 헤매지 말 것.
- CLI가 안 잡히면 PATH부터 확인(예: `/opt/hermes/bin`).

프레임워크의 LLM 라우팅이 경량 모델(기본값인 경우가 많다)로 잡혀 있으면 이 지침의 준수율이 떨어질 수 있다 — AI SPACE 작업 세션은 지침 준수율이 높은 모델로 라우팅하는 것을 권장한다.

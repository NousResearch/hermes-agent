---
name: sinko-sequential-ticket-train-cron
description: Create a batch of Sinko Plane tickets and a cron job that processes exactly one ticket every N minutes using Claude Code, auto-review, PR, and merge without deploying.
---

# sinko-sequential-ticket-train-cron

## 언제 쓰나
- 영재가 `각각 티켓 만들고 10분에 한개씩 처리`, `순서대로 자동 처리`, `차례대로 PR 만들고 자동 리뷰/머지` 같은 요청을 할 때
- 하나의 큰 개선 주제를 여러 개의 작은 Sinko 티켓으로 쪼갠 뒤 **시간 간격을 두고 1개씩 자율 처리**하게 만들고 싶을 때

## 핵심 아이디어
1. 먼저 Plane 티켓을 **명시적으로 여러 개 생성**한다.
2. 각 티켓에 **고정 branch name** 을 박아 둔다.
3. 크론 프롬프트에는 `처리 대상 목록(sequence_id / issue_id / branch / title)` 을 **전부 self-contained** 하게 넣는다.
4. 크론은 매 실행마다 **Done/Cancelled가 아닌 첫 번째 티켓 1개만** 처리하고 종료한다.
5. 구현은 반드시 **Claude Code (`claude -p --dangerously-skip-permissions`)** 를 쓰게 한다.
6. PR 리뷰는 GitHub Action의 `Claude code review`가 자동으로 수행하게 둔다. cron 세션에서 별도 Claude review/ultrareview/requesting-code-review를 실행하지 않는다.
7. PR branch를 push할 때마다 GitHub Action 리뷰가 다시 돌기 때문에, self-healing/fix 작업 중에는 PR review comment에 나온 이슈들을 **하나씩 작업해서 로컬 커밋으로 쌓는다**. 여러 로컬 커밋은 허용하지만, 모든 comment issue 처리 + focused 검증이 끝나기 전까지 remote push 금지. 마지막에 **한 번만 push**해서 PR을 업데이트한다. 여러 번 나눠 push하지 않는다.
8. **배포는 절대 하지 않는다.**

## 왜 이 방식이 필요한가
- cron 세션은 현재 대화 문맥이 없다. 따라서 티켓 목록/브랜치/상태 규칙을 프롬프트 안에 모두 적어야 한다.
- `작업 여러 개를 한 잡이 한 번에 다 처리`하게 두면 실패 시 중간 상태가 엉키기 쉽다.
- `한 번에 1개씩` 처리하면 실패 범위가 작고, 리뷰/머지/티켓 상태 관리가 단순해진다.

## Plane 티켓 생성 패턴
- workspace: `sinko`
- project id: `47641ba5-8621-4661-8538-4bdb5c00fa56`
- 상태는 생성 시 기본 `Backlog`
- 생성 후 응답의 `sequence_id` 로 branch name 확정

브랜치 규칙 예:
- `fix/SINKO-222-upload-request-id-structured-errors`
- `fix/SINKO-223-upload-presign-observed-api`

티켓 description에는 아래 4섹션을 항상 넣는다.
- 요구사항
- 구현 계획
- 영향 범위
- 수락 기준

그리고 patch로 description 맨 위에 branch line을 박는다.
- `Branch: <code>fix/SINKO-222-...</code>`

## 크론 프롬프트에 꼭 넣을 것
아래를 모두 self-contained로 넣어야 한다.

### 1. repo / 금지사항
- repo path
- main 직접 수정 금지
- worktree 필수
- deploy 금지
- pm2 restart 금지
- CHANGELOG 건드리지 말 것

### 2. 툴/방법 강제
- Claude Code 필수
- 구현 작업 기본 호출은 `claude -p --dangerously-skip-permissions` 로 하고, `--max-turns`를 주지 않는다.
- Claude Code 공식 CLI reference 기준 `--max-turns`는 print mode 전용 Claude Code 옵션이며, "No limit by default"다: https://code.claude.com/docs/en/cli-reference
- 로컬 `claude --help` 출력에는 버전에 따라 `--max-turns`가 안 보일 수 있으므로, 필요하면 공식 docs 또는 `claude -p '...' --max-turns 1 --output-format json` smoke로 확인한다.
- 낮은 `--max-turns 6/8/12/15` cap은 Sinko ticket-train 구현 작업에 쓰지 않는다. 단순 diff review/짧은 분석처럼 명백히 작은 작업에만 제한값을 명시할 수 있다.
- runaway 방지는 Claude turn cap이 아니라 cron/terminal timeout, 작업트리 상태 확인, 실패 시 Plane 댓글/보고로 제어한다.
- 로컬 테스트 필수
- 별도 로컬 AI 리뷰 금지: PR 리뷰는 GitHub Action `Claude code review`에 맡김
- PR 업데이트는 검증 후 single push: push마다 리뷰가 돌기 때문에 중간 push 금지
- GitHub Action 리뷰/check가 green일 때만 merge

### 3. Plane 상태/프로젝트 정보
- workspace
- project id
- API key header
- Backlog / In Progress / Done / Cancelled state id

### 4. 처리 대상 목록 / 실행 파라미터
각 티켓마다 다음 4개를 넣는다.
- `sequence_id`
- `issue_id`
- `branch`
- `title`

그리고 train 전체 파라미터를 명시한다.
- `cadence`: 사용자 지정 실행 주기 (`every 30m` 등)
- `review_cycles`: **PR review/self-healing cycle 수**다. cron 실행 횟수/repeat 값이 아니다.
- `repeat`: 숫자로 제한하지 않는다. ticket train은 `forever`로 두고, 남은 티켓이 없고 train 관련 열린 PR/작업도 전혀 없을 때만 완료 조건에서 자기 pause/stop한다.

예:
- `#222 / issue_id 0278... / branch fix/SINKO-222-... / title Add request-id ...`

### 5. 선택 로직
- Plane API로 현재 상태 조회
- Done/Cancelled가 아닌 첫 번째 티켓 선택
- 이미 merged PR 있으면 skip 후 다음 티켓
- 이번 실행에서는 **1개만 처리 후 종료**
- 전부 끝났으면 짧게 완료 보고

### 6. 구현 절차
권장 순서:
1. 티켓 In Progress
2. `scripts/worktree-start.sh <branch>`
3. 관련 파일 읽고 짧은 진단/계획
4. Claude Code로 구현
5. focused tests + `pnpm type-check`
6. GitHub Action 리뷰/check 결과를 기다림. cron 세션에서 별도 로컬 AI 리뷰/ultrareview/requesting-code-review는 실행하지 않음
7. blocker가 있으면 failed logs/review comment를 수집하고, PR review comment에 쓰인 이슈를 하나씩 처리하면서 로컬 커밋을 여러 개 쌓은 뒤 재검증한다. 모든 이슈 처리와 focused 검증이 끝나기 전까지 push하지 않는다.
8. 의도한 파일만 add
9. commit
10. 검증 완료 후 single push
11. gh PR create 또는 기존 PR 업데이트
12. blocker 없으면 gh merge
13. main checkout `git pull --ff-only`
14. Plane Done
15. 한국어 결과 보고

### 6.1 실패 시 Plane 댓글 필수
- 구현/PR 생성/PR 업데이트/merge/self-heal 중 실패하면, 최종 보고 전에 **해당 Plane 티켓에 실패 분석 댓글을 남긴다**.
- 특히 Claude Code가 `error_max_turns`, timeout, provider error로 종료했거나, 종료 후 `git status` / `git diff --stat` 기준 변경이 없으면 반드시 댓글을 남긴다.
- 댓글에는 아래를 포함한다:
  - 시도한 작업과 branch/worktree
  - 정확한 실패 이유 (`error_max_turns`, timeout, failing check, merge conflict 등)
  - Claude max-turn 값과 session id가 출력에 있으면 기록
  - `git status` / `git diff --stat` 결과: 변경 없음인지, partial diff가 있었는지
  - Plane 상태를 Todo로 되돌렸는지 / worktree cleanup을 했는지 / PR이 남아있는지
  - 다음 권장 조치(예: 더 작은 prompt로 재시도, 티켓 분할, 특정 failing check 수정)
- Plane comment endpoint:
  ```bash
  curl -s -X POST "http://127.0.0.1:8080/api/v1/workspaces/sinko/projects/<PROJECT_ID>/issues/<ISSUE_ID>/comments/" \
    -H "X-API-Key: $PLANE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"comment_html":"<div><p>실패 분석...</p></div>"}'
  ```
- POST 응답 또는 comments list로 댓글 생성 여부를 검증한다. 댓글 생성이 401/403/400 등으로 실패하면 그 사실도 최종 보고에 포함한다.

## cronjob 생성 권장 파라미터
사용자에게 train 단위 입력으로 받거나, 명시가 없으면 기본값을 적용한다.

- `cadence` / `schedule`: `"every 10m"`, `"every 30m"`, `"every 1h"`, cron expression 등. 기본 권장값은 `every 30m`; CI/review가 끝날 시간을 고려해 너무 짧게 잡지 않는다.
- `review_cycles`: **PR review/self-healing cycle 수**다. cron 실행 횟수/repeat 값이 아니다. 한 PR에서 Claude/GitHub review 피드백을 반영해 고치는 최대 사이클 수로 해석한다.
- `repeat`: 숫자로 제한하지 않는다. ticket train은 `forever`로 두고, **남은 티켓이 없고 train 관련 열린 PR/작업도 전혀 없을 때만** 완료 조건에서 자기 pause/stop하도록 프롬프트에 명시한다.
- `deliver: origin`
- 생성 직후 `cronjob(action='run', job_id=...)` 으로 **첫 실행 즉시 트리거** 가능
- 사용자가 `다 완료한 다음에는 크론잡 멈춤` / `끝나면 stop/pause` 를 요청하면, 프롬프트에 완료 후 자기 자신 pause 규칙을 명시한다:
  - 완료 조건: train 대상 티켓이 모두 Done/Cancelled이고 train 관련 열린 구현 PR이 없음
  - 완료 조건을 확인한 실행에서 최종 완료 요약을 작성한 뒤 `cronjob(action="pause", job_id="<JOB_ID>")` 호출
  - 새 cron job 생성 금지
  - pause 실패 시 job_id와 실패 이유를 최종 보고에 포함

## 실전에서 유용했던 세부 규칙
- PR-gated train에서는 실행 시작 시 main 상태/fetch/open PR을 먼저 확인한다. train 범위 구현 PR이 이미 열려 있으면 새 티켓을 시작하지 않는다. 해당 PR이 draft가 아니고 모든 required/status checks가 `COMPLETED/SUCCESS`이며 `mergeable == MERGEABLE`이면 `gh pr merge <num> --squash --delete-branch`로 먼저 merge하고, main checkout에서 `git pull --ff-only` 한 뒤 PR title/body/branch에서 SINKO 번호를 찾아 Plane 티켓을 `Done`으로 PATCH한다. 그 후 같은 실행에서 다음 Todo 티켓 1개까지 시작할 수 있다.
- 실패한 PR을 이어서 고칠 때는 **반드시 최신 GitHub Action Claude review comment를 다시 가져온다**. 이전 세션/이전 SHA의 review 이슈가 이미 해결되어 있고 새 SHA에서 다른 blocker로 바뀌는 경우가 있다. `gh api repos/<owner>/<repo>/issues/<pr>/comments --paginate --jq '[.[] | select(.body|contains("Claude PR Review"))][-1].body'` 처럼 최신 comment를 저장/읽고, `gh pr view <pr> --json headRefOid,statusCheckRollup,mergeable` 로 head SHA와 check 상태를 함께 확인한다.
- 중요한 설계 선택: PR-gated train의 기본은 **merge-and-continue gate**다. 열린 구현 PR이 있으면 먼저 checks/mergeability를 확인한다. 모든 required/status checks가 `SUCCESS`이고 `mergeable == MERGEABLE`이면 그 PR을 squash merge하고 Plane 티켓을 Done으로 바꾼 뒤, **같은 실행에서 다음 Todo 티켓 1개까지 진행**한다. checks가 진행 중/실패/없음/merge 불가이면 새 티켓을 시작하지 않고 대기/실패 보고 후 종료한다. 영재가 `체크 페일이면 수정해서 PR update` 같은 동작을 기대/요구하면 프롬프트에 별도 **self-healing failure loop**를 명시해야 한다.
- self-healing failure loop를 넣을 때는 안전장치를 같이 박는다: failing open PR이 있으면 새 티켓 금지 → `gh pr view`/failed run logs/Claude review comment로 원인 수집 → 기존 PR branch/worktree 재사용 또는 worktree 생성 → Claude Code로 해당 티켓/PR 범위만 수정. **PR review comment에 쓰인 이슈들은 하나씩 작업해서 각각 로컬 커밋으로 만들고, 여러 커밋을 로컬에 쌓은 다음, focused 검증/typecheck/build/agent-test까지 끝낸 뒤 마지막에 한 번만 remote PR branch로 push한다.** push마다 PR review action이 돌기 때문에, comment issue 하나 처리할 때마다 push하면 안 된다. 리뷰/수정 사이클은 PR당 최대 `review_cycles` 회까지 허용한다(기본 4, 사용자 override 가능). cap 이후에도 green이 아니면 정확한 blocker와 시도 횟수를 보고하고 중단한다. merge conflict나 unsafe/broad change가 필요하면 자동 수정하지 말고 보고 후 중단한다.
- Public API v1처럼 설계 문서 PR이 구현 PR과 병렬로 열려 있을 수 있는 train에서는 blocker 제외 PR을 명시한다. 예: PR #812 `docs: add public API ERD`는 구현 PR blocker가 아니며 필요하면 `origin/docs/api-erd:docs/feature/api/erd.md`만 참고한다.
- Claude Code가 timeout/interrupt(`124`, `130`) 또는 provider/tool error로 끝나면, 다른 에이전트로 바꾸지 말고 작업트리 상태를 먼저 본다. 구현 호출에는 기본적으로 `--max-turns`를 주지 않는다. `git status --short`/`git diff --stat`가 비어 있고 작업이 여전히 유효하면, 같은 Claude Code를 훨씬 작고 명령형인 프롬프트로 1회 재시도한다. 프롬프트에는 exact files to create/edit, no broad searches, no dev server, stop after writing 같은 제약을 넣는다. 정말 작은 follow-up/review가 아니라면 낮은 6/8/12/15 turn cap을 붙이지 않는다.
- Claude Code가 timeout/interrupt(`124`, `130`)로 끝나면 **출력/프롬프트를 믿지 말고 작업트리 상태를 source of truth로 본다**. 즉시 `git status --short --branch`, `git diff --stat`, 관련 파일 diff를 확인하고, 어떤 파일이 실제로 dirty인지 기준으로 다음 조치를 정한다. 이전 세션 요약의 dirty 파일 목록과 현재 상태가 다를 수 있다.
- PR review fix loop에서 partial local changes가 생기면 먼저 review blocker와 현재 diff를 매칭한다. blocker와 무관하거나 conceptual risk가 있는 변경(예: concurrency TOCTOU를 `Date.now()/1000` 같은 약한 workaround로 덮는 변경)은 유지하지 말고 더 단순하고 검증 가능한 방향으로 되돌린다.
- SQL/text-regression test를 Claude로 고친 뒤에는 반드시 focused test를 직접 재실행한다. 리뷰 대응 중 SQL 주석을 추가하면 기존 정규식 검증(`ORDER BY ... LIMIT` 등)이 주석 때문에 깨질 수 있다. 이 경우 구현을 되돌리기보다 테스트가 executable SQL만 보도록 line comment를 제거한 문자열(`sql.replace(/--[^\n]*/g, "")`) 기준으로 검증하게 하는 작은 follow-up 패치는 허용된다.
- train PR 생성 시 `scripts/worktree-done.sh`는 편하지만 내부에서 `git add -A`를 사용하고 PR body도 고정이라, `git add . 금지`/한국어 PR body/검증 결과/리스크 포함 요구가 있는 cron에서는 수동 flow를 선호한다: focused 검증 → independent review → 의도 파일만 `git add <file...>` → 필요 시 package version bump도 명시적으로 add → commit → rebase origin/main → push → `gh pr create --body-file`.
- 프롬프트에 `새로운 크론을 만들지 말 것` 을 넣어 recursion 방지
- `각 실행에서 티켓 1개만 처리` 를 반복해서 명시
- `이미 대응 PR이 열려 있거나 머지되었는지도 gh로 확인` 규칙을 넣으면 중복 처리 방지에 도움됨
- PR-only train(자동 merge 금지)에서는 실행 시작 전에 `gh pr list --state open --json number,title,headRefName,body,url` 로 train 범위 구현 PR blocker를 먼저 찾는다. title/body/branch에 train 키워드나 티켓 번호가 있는 열린 구현 PR이 있으면 새 티켓을 시작하지 않고 해당 PR URL만 보고한다. 단, 설계/문서 전용 PR을 blocker에서 제외해야 하는 경우(예: Public API v1에서 `docs: add public API ERD` #812) 예외 PR 번호/제목을 프롬프트에 명시한다.
- PR-only train에서는 PR 생성 후 Plane 티켓은 `In Progress`로 유지하고, PR URL을 Plane comment로 남긴다. 이후 실행에서 해당 PR이 merge된 것을 확인했을 때만 티켓을 `Done`으로 바꾼다.
- `배포는 안 함` / `pm2 restart 안 함` / `main merge 안 함` 을 분명히 적어야 cron이 과하게 나가지 않음
- 문서/계약 전용 티켓도 focused validation을 둔다. 예: OpenAPI YAML은 `node --test scripts/test/api-openapi.test.mjs`, Markdown은 `npx --yes markdownlint-cli2 <file>`.
- 문서에 API key/token 예시를 쓸 때 Hermes redaction이 `<raw_key>`/credential-looking placeholder를 `***`로 바꿀 수 있다. OpenAPI/Markdown에는 실제 key 형태나 credential-looking placeholder 대신 “HTTP Authorization header 기반 bearer token” 같은 중립 문구를 사용하고, 디스크에서 다시 읽어 깨진 문장이 없는지 확인한다.
- `한국어로 보고, 코드/PR/브랜치/주석은 영어 유지` 같은 출력 규칙도 함께 박아 두면 결과물이 안정적임

## 티켓 일괄 생성 팁
파이썬/urllib로 Plane API를 때릴 때는:
- 먼저 issue 생성
- 응답에서 `sequence_id` 추출
- branch name 계산
- PATCH로 description_html 업데이트

이 과정을 스크립트 1회로 돌리면 여러 티켓을 빠르게 만들 수 있다.

## 주의사항
- cron 세션은 사용자 질문을 못 한다. 애매한 부분이 남지 않게 목록/규칙을 프롬프트에 다 넣어야 한다.
- ticket train cron은 숫자 `repeat`로 소진시키지 않는다. `repeat: forever`를 사용하고, 남은 티켓이 없고 train 관련 열린 PR/작업도 전혀 없을 때만 완료 조건에서 자기 pause/stop하도록 만든다.
- merge 후에도 deploy를 자동으로 이어서 하면 안 된다. 영재 shorthand `마무리`와는 별개다.
- branch 이름은 티켓 생성 후 `sequence_id` 를 받은 다음 확정해야 한다.
- cron job이 `RuntimeError: Unknown provider 'openai'` 로 즉시 실패하면, job의 pinned provider가 현재 런타임 provider와 안 맞는 것이다. `cronjob(action='update', job_id=..., model={'provider':'openai-codex','model':'gpt-5.5'})` 처럼 현재 사용 가능한 provider로 업데이트한 뒤 즉시 `cronjob(action='run', ...)` 으로 재검증한다.
- `cronjob(action='run')` 는 실행 예약만 걸 수 있으니, 바로 성공으로 단정하지 말고 `~/.hermes/cron/output/<job_id>/` 최신 `.md` 파일 또는 `cronjob(action='list')` 의 `last_status` 를 확인한다. 최신 output의 `## Response` 또는 `## Error` 를 읽어 실제 결과를 검증한다.

## 결과 보고 포맷 권장
- 이번에 처리한 티켓 번호/제목
- 변경 파일
- 테스트 명령과 결과
- PR 링크
- merge 여부
- 남은 티켓 번호 목록

## 재사용 포인트
이 스킬은 observability 작업뿐 아니라,
- admin cleanup
- export fixes
- upload follow-ups
- analytics hygiene
처럼 **비슷한 성격의 작은 티켓 여러 개를 순차 자동 처리**할 때 그대로 재사용할 수 있다.

# Operations

## 설치

```bash
cd ~/.hermes/hermes-agent
hegi/scripts/install.sh
```

설치기는 실제 Hermes config, `.env`, session DB에서 다음 항목을 자동 탐지한다.

- Memory Forest curator MCP가 설정된 profile
- `HEGI_GROUP_CHAT_ID` 또는 실제 group session의 Telegram chat ID
- Telegram allowlist/home channel의 교수 user ID
- 해당 chat을 가진 HeHe, HeCo, HeClaude DB
- Memory Forest의 기본 STM project

기존 HEGI config는 timestamped backup으로 보존한다. profile-local
`hegi-telegram` plugin을 설치·활성화하고, 실행 중인 curator gateway가 있으면
안전하게 재시작한다. YAML 수동 편집은 정상 설치 절차가 아니다.

WSL에서 user systemd를 사용할 수 없으면 Windows Startup의
`Hermes-HEGI-v2.cmd`를 등록한다. 일반 Linux에서는 user systemd unit을 활성화한다.

```bash
python -m hegi doctor
python -m hegi run-once
```

`run-once`와 `daemon`은 기본적으로 Telegram dry-run이다. 실제 송신은 명시적으로
`--send`를 붙인 경우에만 수행한다.

```bash
python -m hegi run-once --send
hegi/scripts/start.sh --send
python -m hegi status
```

`start.sh`은 10초 동안 `daemon.ready`를 확인한 뒤에만 성공한다. daemon은
`daemon.lock`의 `flock`과 PID identity 검사로 중복 실행 및 PID 재사용을 막는다.
비정상 종료 뒤 stale PID가 남아도 다음 시작에서 복구한다.

## Telegram 승인과 Draft

교수가 회의록에 `기억해` 또는 `초안 만들어`라고 답하면 Telegram
`pre_gateway_dispatch` plugin이 메시지를 일반 agent turn보다 먼저 선점한다. 회의록
Telegram message ID로 meeting을 찾고, 교수 ID와 중복 message ID를 검증한 뒤
approval job을 영속화한다. gateway나 daemon이 재시작되어도 pending job은 다시
처리된다.

운영 복구 시에만 다음 CLI로 같은 gate를 직접 호출할 수 있다.

```bash
python -m hegi approve \
  --meeting-id HEGI_MEETING_ID \
  --text '기억해' \
  --user-id TELEGRAM_USER_ID \
  --message-id TELEGRAM_MESSAGE_ID
```

`--project`를 생략하면 자동 탐지된 `memory.default_project`를 사용한다. 두 경로 모두
Memory Forest를 다시 검색한 뒤 pending STM Draft만 만든다. Commit은 없다.

## 중지와 롤백

```bash
hegi/scripts/stop.sh
git revert HEGI_COMMIT
```

상태, approval queue와 archive를 보존한 채 daemon만 중지한다. Windows 자동 시작을
영구 제거하려면 Startup 폴더의 `Hermes-HEGI-v2.cmd`도 제거한다. v1 복귀 절차는
[migration.md](migration.md)에 있다.

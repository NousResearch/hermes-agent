# Operations

## 설치

```bash
cd ~/.hermes/hermes-agent
hegi/scripts/install.sh
$EDITOR ~/.hermes/profiles/memory-curator/hegi/config.yaml
```

Draft MCP를 사용하려면 memory-curator profile의 `HERMES_HOME`에서 실행한다.

```bash
export HERMES_HOME="$HOME/.hermes/profiles/memory-curator"
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

## 승인과 Draft

Telegram 승인 event의 교수 user ID와 message ID를 gateway integration에서 전달한다.
수동 운영 시 다음 명령이 같은 gate를 사용한다.

```bash
python -m hegi approve \
  --meeting-id HEGI_MEETING_ID \
  --text '@헤기 초안 만들어' \
  --user-id TELEGRAM_USER_ID \
  --message-id TELEGRAM_MESSAGE_ID \
  --project media_aesthetics
```

이 명령은 Memory Forest를 다시 검색한 뒤 pending STM Draft만 만든다. Commit은 없다.

## 중지와 롤백

```bash
hegi/scripts/stop.sh
git revert HEGI_COMMIT
```

상태와 archive를 보존한 채 daemon만 중지한다. v1 복귀 절차는
[migration.md](migration.md)에 있다.

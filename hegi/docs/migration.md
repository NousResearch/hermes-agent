# v1 Migration

`migrate_v1.sh`은 기본 실행 시 진단만 출력한다. `--apply`를 주면 PID file과 실제
command line을 함께 검증하여 기존 v1 loop를 종료하고, v1 state와 기존 v2 config를
timestamped backup으로 보존한다. 이어서 환경 자동 탐지, v2 활성화, Telegram
pre-dispatch plugin 설치와 gateway 반영, HEGI daemon `--send` 시작까지 수행한다.
기존 watcher 파일과 Git stash는 삭제하거나 변경하지 않는다.

```bash
hegi/scripts/migrate_v1.sh
hegi/scripts/migrate_v1.sh --apply
```

명령이 성공하면 `daemon.ready`가 생성되어 있고 `python -m hegi status`의
`daemon.alive`가 true이다. 롤백 시 HEGI daemon을 중지하고 profile config backup을
복원한 뒤 기존 `~/bin/hegi-memory-watch-loop`를 다시 실행하면 된다. 두 감시기를
동시에 실제 Telegram 송신 모드로 실행하지 않는다.

# v1 Migration

`migrate_v1.sh`은 기본 실행 시 진단만 출력한다. `--apply`를 주면 기존 v1 loop에
TERM을 보내고 v1 state를 timestamped backup으로 보존한 뒤 HEGI config를 설치한다.
기존 파일과 stash는 삭제하거나 변경하지 않는다.

```bash
hegi/scripts/migrate_v1.sh
hegi/scripts/migrate_v1.sh --apply
```

HEGI를 검증한 뒤 daemon을 시작한다. 롤백 시 HEGI daemon을 중지하고 기존
`~/bin/hegi-memory-watch-loop`를 다시 실행하면 된다. 두 감시기를 동시에 실제
Telegram 송신 모드로 실행하지 않는다.

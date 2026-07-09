# Changelog

## v0.4.0 - 2026-07-10

### Added

- Telegram trusted group mention/reply model route
  - 허용된 group/supergroup에서 봇 mention 또는 bot reply에만 응답
  - 일반 group 메시지는 무시하고 private DM route와 분리
- group 전용 session
  - `telegram:trusted_group:<chat_id>:<generation_id>`
  - `telegram:trusted_forum:<chat_id>:<thread_id>:<generation_id>`
  - idle timeout 후 새 generation, 기본 idle timeout 120분
- group route model override
  - `TELEGRAM_GROUP_QA_MODEL`
  - `TELEGRAM_GROUP_QA_REASONING_EFFORT`
  - trusted friend group에서 `gpt-5.6-luna`와 `low` reasoning effort 사용 가능
- trusted group read-only context
  - active day, meals, last meal, outings, work, todos, HRT read-only 요약

### Changed

- group command menu를 빈 목록으로 정리
- private DM command menu는 유지
- slash command 중심 UX에서 mention/reply 중심 UX로 전환

### Security / Privacy

- group route tools를 빈 목록으로 강제해 mutation/write/check-in/check-out/project edit 차단
- spark/private spark를 model 호출 전에 pre-filter로 차단
- token, credential, `.env`, DM/message-log 원문, 계좌·세금·정확한 주소 요청 차단
- HRT raw note 제외

### Tests

- trusted group Q&A/model route
- session isolation과 idle generation reset
- model override와 spark pre-filter

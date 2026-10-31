# 야간 비즈니스 fallback (night_business_fallback)

야간 시간대에 개인 계정 quota가 소진됐을 때, 별도의 비즈니스 프로파일로
Kanban 워커를 투명하게 라우팅하는 선택적 기능이다.
**기본값은 `enabled: false`** — 명시적으로 활성화해야 동작한다.

---

## 설정 키

`config.yaml`의 `kanban.night_business_fallback` 섹션:

```yaml
kanban:
  night_business_fallback:
    enabled: false          # 기본값. true로 설정해야 활성화.
    timezone: "Asia/Seoul"  # 창(window) 판정 기준 시간대. 시스템 TZ·DST 무관.
    start_time: "21:00"     # 야간 창 시작 (포함). HH:MM 형식.
    end_time: "06:00"       # 야간 창 종료 (미포함). 자정을 넘길 수 있음.
    profiles:
      anthropic: "night-business-claude"   # Anthropic 소진 시 사용할 프로파일
      openai-codex: "night-business-codex" # Codex 소진 시 사용할 프로파일
```

### 키 설명

| 키 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `enabled` | bool | `false` | `true`일 때만 라우팅이 동작함 |
| `timezone` | str | `"Asia/Seoul"` | 창 판정 시간대. [IANA 이름](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) |
| `start_time` | str | `"21:00"` | 야간 창 시작 (포함), HH:MM |
| `end_time` | str | `"06:00"` | 야간 창 종료 (미포함), HH:MM |
| `profiles.anthropic` | str | (없음) | Anthropic quota 소진 시 사용할 Hermes 프로파일 이름 |
| `profiles.openai-codex` | str | (없음) | OpenAI Codex quota 소진 시 사용할 Hermes 프로파일 이름 |

---

## 시간대 변경 방법

`timezone` 값을 원하는 IANA 시간대 이름으로 교체한다.

```yaml
kanban:
  night_business_fallback:
    timezone: "America/New_York"
    start_time: "22:00"
    end_time: "07:00"
```

변경은 다음 dispatcher tick부터 즉시 반영된다 (config는 매번 새로 읽음).
시스템 환경변수 `TZ`나 DST는 판정에 영향을 미치지 않는다.

---

## 동작 원리

1. dispatcher가 task를 spawn하기 직전 개인 프로파일의 quota를 확인한다.
2. 현재 시각이 야간 창 내에 있고 개인 quota가 소진된 경우에만 비즈니스 프로파일로 전환한다.
3. task의 `assignee`는 변경되지 않는다 (감사 추적 보존).
4. 비즈니스 프로파일로 spawn하면 DB에 `night_business_route` 이벤트가 기록된다.

### fail-closed 정책

다음 상황에서는 비즈니스 라우팅을 하지 않는다 ("판정불가"):

- 개인 credential pool에 자격증명 자체가 없음 (`personal_unknown`)
- usage API가 429/401을 반환해 가용 여부를 판단할 수 없음 → 내부 예외 → deny
- 비즈니스 프로파일의 quota도 소진됨 (`business_unavailable`)
- `profiles` 맵에 해당 provider가 없음 (`business_profile_missing`)
- 야간 창 외부 (`outside_window`)

---

## 비즈니스 프로파일 등록 절차

비즈니스 계정용 별도 Hermes 프로파일을 생성하고 인증 정보를 등록한다.

```bash
# 1. 프로파일 생성 (처음 한 번)
hermes -p night-business-claude profile create

# 2. Anthropic OAuth 자격증명 추가
hermes -p night-business-claude auth add

# 3. 동일하게 Codex용도 등록 (필요한 경우)
hermes -p night-business-codex profile create
hermes -p night-business-codex auth add
```

등록 후 `hermes config set kanban.night_business_fallback.enabled true`로 활성화한다.

---

## 주의사항

- 비즈니스 프로파일의 `auth.json`은 개인 credential pool과 완전히 격리된다.
  dispatcher는 각 프로파일을 별도의 `HERMES_HOME` 컨텍스트에서 로드한다.
- 이 기능은 dispatcher tick에서만 동작한다. 실행 중인 worker 세션에는 영향을 주지 않는다.
- Discord 알림(`#업무-진행`) 연동은 NF-2 이후 별도 설정이 필요하다.

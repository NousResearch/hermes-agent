# Hermes Docs — Local Mac Launcher

이 문서는 Hermes Docs 워크스페이스를 로컬 웹 앱으로 실행하는 방법을 설명합니다.
DMG 없이, 별도 의존성 없이, Hermes가 설치된 모든 Mac에서 즉시 실행됩니다.

---

## 사전 조건

Hermes가 설치되어 있어야 합니다.

```bash
# Hermes가 없다면 설치
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

설치 후 `hermes` 명령이 PATH에 있는지 확인합니다.

```bash
hermes --version
```

---

## 실행 방법

### 가장 간단한 방법

```bash
bash scripts/launch-hermes-docs.sh
```

브라우저가 자동으로 `http://localhost:9119/docs-workspace`를 엽니다.

### 옵션

| 옵션 | 설명 |
|---|---|
| `--port <번호>` | 포트 지정 (기본값: 9119) |
| `--no-open` | 브라우저 자동 실행 안 함 — URL만 출력 |
| `--help` | 도움말 출력 |

```bash
# 포트 변경
bash scripts/launch-hermes-docs.sh --port 8787

# 브라우저 자동 실행 없이 URL만 출력
bash scripts/launch-hermes-docs.sh --no-open

# 둘 다
bash scripts/launch-hermes-docs.sh --port 8787 --no-open
```

---

## URL 구조

| 목적지 | URL |
|---|---|
| Docs 탭 (직접 이동) | `http://localhost:<port>/docs-workspace` |
| 대시보드 홈 | `http://localhost:<port>` |

스크립트는 항상 두 URL 모두 터미널에 출력합니다.

---

## 종료

터미널에서 `Ctrl-C`를 누르면 서버가 멈춥니다.  
백그라운드에서 실행 중인 대시보드를 끄려면:

```bash
hermes dashboard --stop
```

---

## 작동 원리

`launch-hermes-docs.sh`는 다음 순서로 동작합니다.

1. `hermes`가 PATH에 있는지 확인합니다. 없으면 설치 안내를 출력하고 종료합니다.
2. `hermes dashboard --port <port> --no-open`을 실행합니다.
3. 서버가 준비될 때까지 1.5초 대기 후 `/docs-workspace` URL을 브라우저로 엽니다 (`--no-open`이 없을 때).

사용자 자격증명, 프로필, 워크스페이스 설정은 일절 수정하지 않습니다.

---

## 관련 문서

- 전체 패키징 계획 (DMG 포함): `docs/superpowers/plans/2026-05-11-hermes-docs-mac-packaging-plan.md`
- Hermes Docs 플러그인: `plugins/hermes-docs/`

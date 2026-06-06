<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
</p>

**[Nous Research](https://nousresearch.com)에서 개발한 스스로 학습하고 발전하는(self-improving) AI 에이전트.**
Hermes Agent는 경험을 통해 스스로 기술을 생성하고, 사용 과정에서 이를 개선하며, 스스로 지식을 유지하도록 상기시키고, 과거 대화를 검색하며, 여러 세션에 걸쳐 사용자에 대한 깊이 있는 이해 모델을 구축하는 학습 루프가 내장된 유일한 에이전트입니다. 월 $5 수준의 VPS, GPU 클러스터, 또는 대기 상태일 때 거의 비용이 들지 않는 서버리스 인프라 등 어디서나 실행할 수 있습니다. 노트북 사양에 구애받지 않으며, 에이전트가 클라우드 VM에서 작동하는 동안 사용자는 Telegram으로 대화할 수 있습니다.

원하는 모든 모델을 지원합니다 — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200개 이상의 모델), [NovitaAI](https://novita.ai) (모델 API, 에이전트 샌드박스, GPU 클라우드를 지원하는 AI 네이티브 클라우드), [NVIDIA NIM](https://build.nvidia.com) (Nemotron), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI 또는 자체 엔드포인트. 코드 변경이나 락인(lock-in) 없이 `hermes model` 명령어로 자유롭게 전환해 보세요.

<table>
<tr><td><b>진짜 터미널 인터페이스</b></td><td>멀티라인 편집, 슬래시(/) 명령어 자동 완성, 대화 기록, 중단 및 리디렉션, 스트리밍 도구 출력을 지원하는 완벽한 TUI.</td></tr>
<tr><td><b>사용자가 있는 곳 어디나</b></td><td>단일 게이트웨이 프로세스 하나로 Telegram, Discord, Slack, WhatsApp, Signal, CLI 모두 지원. 음성 메모 전사, 크로스 플랫폼 대화 연속성 보장.</td></tr>
<tr><td><b>닫힌 학습 루프 (Closed Loop)</b></td><td>주기적인 자극(nudge)을 통해 에이전트가 직접 관리하는 메모리. 복잡한 작업 수행 후 자율적인 기술 생성. 사용 과정에서의 기술 스스로 개선. 크로스 세션 리콜을 위한 LLM 요약 결합 FTS5 세션 검색. <a href="https://github.com/plastic-labs/honcho">Honcho</a> 변증법적 사용자 모델링. <a href="https://agentskills.io">agentskills.io</a> 개방형 표준 호환.</td></tr>
<tr><td><b>예약 자동화</b></td><td>모든 플랫폼으로 전송이 가능한 내장 cron 스케줄러. 일일 보고서, 야간 백업, 주간 감사 등 모든 작업을 자연어로 설명하여 무인 실행 가능.</td></tr>
<tr><td><b>위임 및 병렬 처리</b></td><td>병렬 워크플로우를 처리하기 위해 격리된 서브 에이전트 생성. RPC를 통해 도구를 호출하는 Python 스크립트를 작성하여 여러 단계의 파이프라인을 컨텍스트 비용이 전혀 없는 턴으로 압축.</td></tr>
<tr><td><b>내 노트북뿐만 아니라 어디서든 실행</b></td><td>로컬, Docker, SSH, Singularity, Modal, Daytona 등 6가지 터미널 백엔드 지원. Daytona와 Modal은 서버리스 지속성을 제공하여, 사용하지 않을 때는 에이전트 환경이 휴면 상태가 되고 필요할 때 활성화되어 세션 사이에 비용이 거의 발생하지 않음. $5 VPS 또는 GPU 클러스터에서도 실행 가능.</td></tr>
<tr><td><b>연구 준비 완료</b></td><td>차세대 도구 호출(tool-calling) 모델 학습을 위한 배치 트래젝토리(trajectory) 생성 및 트래젝토리 압축.</td></tr>
</table>

---

## 빠른 설치

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (네이티브, PowerShell)

> **주의:** 네이티브 Windows는 WSL 없이 Hermes를 실행합니다 — CLI, 게이트웨이, TUI 및 도구 모두 네이티브로 작동합니다. WSL2를 사용하고 싶다면, 위의 Linux/macOS용 한 줄 명령어를 그대로 사용할 수도 있습니다. 버그를 발견하셨나요? [이슈 등록](https://github.com/NousResearch/hermes-agent/issues)을 해주세요.

PowerShell에서 다음 명령을 실행합니다:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

설치 프로그램이 uv, Python 3.11, Node.js, ripgrep, ffmpeg 및 **포터블 Git Bash** (MinGit, `%LOCALAPPDATA%\hermes\git` 경로에 압축 해제 — 관리자 권한 불필요, 기존 시스템의 Git 설치와 완전히 격리됨)를 포함한 모든 환경을 알아서 처리합니다. Hermes는 셸 명령어를 실행할 때 이 번들링된 Git Bash를 사용합니다.

시스템에 이미 Git이 설치되어 있는 경우, 설치 프로그램이 이를 감지하여 사용합니다. 그렇지 않은 경우 약 45MB 크기의 MinGit을 다운로드하는 것만으로 충분하며, 시스템의 다른 Git 환경을 전혀 건드리지 않고 방해하지도 않습니다.

> **Android / Termux:** 테스트된 수동 설치 경로는 [Termux 가이드](https://hermes-agent.nousresearch.com/docs/getting-started/termux)에 문서화되어 있습니다. Termux 환경에서는 전체 `.[all]` 익스트라가 Android와 호환되지 않는 음성 관련 종속성을 가져오기 때문에, 엄선된 `.[termux]` 익스트라를 설치합니다.
>
> **Windows:** 네이티브 Windows를 완전히 지원하며, 위의 PowerShell 한 줄 명령어로 모든 것을 설치할 수 있습니다. WSL2를 사용하고자 하신다면 Linux 명령어도 동일하게 작동합니다. 네이티브 Windows 설치본은 `%LOCALAPPDATA%\hermes`에 저장되며, WSL2의 경우 Linux처럼 `~/.hermes` 아래에 설치됩니다. 현재 브라우저 기반 대시보드 채팅 창(POSIX PTY 사용 — 클래식 CLI 및 게이트웨이는 모두 네이티브로 실행됨) 기능을 사용하기 위해서만 특별히 WSL2가 필요합니다.

설치 완료 후:

```bash
source ~/.bashrc    # 셸 재로드 (또는: source ~/.zshrc)
hermes              # 대화 시작!
```

---

## 시작하기

```bash
hermes              # 대화식 CLI — 대화 시작
hermes model        # LLM 제공업체 및 모델 선택
hermes tools        # 활성화할 도구 구성
hermes config set   # 개별 설정 값 설정
hermes gateway      # 메시징 게이트웨이 시작 (Telegram, Discord 등)
hermes setup        # 전체 설정 마법사 실행 (모든 설정을 한 번에 구성)
hermes claw migrate # OpenClaw에서 마이그레이션 (OpenClaw에서 전환하는 경우)
hermes update       # 최신 버전으로 업데이트
hermes doctor       # 문제 진단
```

📖 **[전체 문서 →](https://hermes-agent.nousresearch.com/docs/)**

---

## API 키 수집 없이 바로 사용하기 — Nous Portal

Hermes는 사용자가 원하는 모든 제공업체를 지원하며, 이는 변하지 않습니다. 하지만 모델, 웹 검색, 이미지 생성, TTS, 클라우드 브라우저용 API 키를 각각 5개씩 따로 수집하고 관리하고 싶지 않다면, **[Nous Portal](https://portal.nousresearch.com)** 구독 하나로 이 모든 것을 해결할 수 있습니다:

- **300개 이상의 모델** — `/model <이름>` 명령어로 언제든 선택 가능
- **도구 게이트웨이 (Tool Gateway)** — 웹 검색 (Firecrawl), 이미지 생성 (FAL), 텍스트 음성 변환 (TTS, OpenAI), 클라우드 브라우저 (Browser Use) 모두 구독을 통해 라우팅됩니다. 별도의 계정을 만들 필요가 없습니다.

새로 설치한 후 아래 명령어 하나면 충분합니다:

```bash
hermes setup --portal
```

이 명령어는 OAuth를 통해 로그인하고, Nous를 제공업체로 설정하며, 도구 게이트웨이를 활성화합니다. 언제든지 `hermes portal info` 명령어로 연결 상태를 확인할 수 있습니다. 자세한 내용은 [도구 게이트웨이 문서 페이지](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway)를 참고하세요.

필요할 때마다 도구별로 본인의 API 키를 사용할 수도 있습니다 — 게이트웨이는 백엔드별로 작동하므로 전부 사용하거나 전부 사용하지 않는 이분법적 방식이 아닙니다.

---

## CLI 대 메시징 빠른 참조

Hermes에는 두 가지 진입점이 있습니다: `hermes` 명령어로 터미널 UI를 시작하거나, 게이트웨이를 실행하여 Telegram, Discord, Slack, WhatsApp, Signal, 이메일로 에이전트와 대화하는 방법입니다. 대화가 시작되면 두 인터페이스 모두에서 많은 슬래시(/) 명령어를 공유하여 사용할 수 있습니다.

| 작업 | CLI | 메시징 플랫폼 |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| 대화 시작 | `hermes` | `hermes gateway setup` + `hermes gateway start` 실행 후 봇에 메시지 전송 |
| 새로운 대화 시작 | `/new` 또는 `/reset` | `/new` 또는 `/reset` |
| 모델 변경 | `/model [provider:model]` | `/model [provider:model]` |
| 페르소나(성격) 설정 | `/personality [이름]` | `/personality [이름]` |
| 이전 턴 다시 시도 또는 실행 취소 | `/retry`, `/undo` | `/retry`, `/undo` |
| 컨텍스트 압축 / 사용량 확인 | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| 기술 찾아보기 | `/skills` 또는 `/<기술-이름>` | `/<기술-이름>` |
| 진행 중인 작업 중단 | `Ctrl+C` 또는 새 메시지 보내기 | `/stop` or 새 메시지 보내기 |
| 플랫폼별 상태 확인 | `/platforms` | `/status`, `/sethome` |

전체 명령어 목록은 [CLI 가이드](https://hermes-agent.nousresearch.com/docs/user-guide/cli) 및 [메시징 게이트웨이 가이드](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)를 참고하세요.

---

## 문서

모든 문서는 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**에서 확인할 수 있습니다:

| 섹션 | 다루는 내용 |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [빠른 시작](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | 설치 → 설정 → 2분 만에 첫 대화 시작하기 |
| [CLI 사용법](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | 명령어, 단축키, 페르소나, 세션 |
| [구성](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 구성 파일, 제공업체, 모델, 모든 옵션 |
| [메시징 게이트웨이](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [보안](https://hermes-agent.nousresearch.com/docs/user-guide/security) | 명령어 승인, DM 페어링, 컨테이너 격리 |
| [도구 및 도구 세트](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40개 이상의 도구, 도구 세트 시스템, 터미널 백엔드 |
| [기술 시스템](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 절차적 메모리, 기술 허브(Skills Hub), 기술 생성 |
| [메모리](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 영구 메모리, 사용자 프로필, 모범 사례 |
| [MCP 통합](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | MCP 서버를 연결하여 에이전트 능력 확장 |
| [Cron 스케줄링](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | 플랫폼 전송을 지원하는 예약 작업 |
| [컨텍스트 파일](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 모든 대화의 형태를 잡아주는 프로젝트 컨텍스트 |
| [아키텍처](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | 프로젝트 구조, 에이전트 루프, 주요 클래스 |
| [기여하기](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | 개발 환경 설정, PR 프로세스, 코드 스타일 |
| [CLI 레퍼런스](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | 모든 명령어 및 플래그 |
| [환경 변수](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | 전체 환경 변수 참조 가이드 |

---

## OpenClaw에서 마이그레이션

기존에 OpenClaw를 사용하고 있었다면, Hermes가 설정, 메모리, 기술 및 API 키를 자동으로 가져올 수 있습니다.

**최초 설정 시:** 설정 마법사(`hermes setup`)가 `~/.openclaw` 디렉토리를 자동으로 감지하고, 구성을 시작하기 전에 마이그레이션 여부를 묻습니다.

**설치 후 언제든지:**

```bash
hermes claw migrate              # 대화형 마이그레이션 (전체 프리셋)
hermes claw migrate --dry-run    # 마이그레이션 대상 미리보기
hermes claw migrate --preset user-data   # 비밀번호/API 키를 제외하고 사용자 데이터만 마이그레이션
hermes claw migrate --overwrite  # 기존 파일과 충돌 시 덮어쓰기
```

가져오는 항목:

- **SOUL.md** — 페르소나 파일
- **메모리** — MEMORY.md 및 USER.md 항목
- **기술** — 사용자가 생성한 기술 → `~/.hermes/skills/openclaw-imports/` 경로로 저장
- **명령어 허용 리스트** — 승인 패턴
- **메시징 설정** — 플랫폼 구성, 허용된 사용자, 작업 디렉토리
- **API 키** — 허용 리스트에 등록된 비밀 키 (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS 자산** — 작업 공간 오디오 파일
- **작업 공간 지침** — AGENTS.md (`--workspace-target` 사용 시)

모든 옵션을 보려면 `hermes claw migrate --help` 명령을 사용하거나, 대화형 마이그레이션 가이드를 지원하는 `openclaw-migration` 기술을 드라이 런(dry-run) 미리보기와 함께 활용해 보세요.

---

## 기여하기

기여는 언제나 환영합니다! 개발 환경 설정, 코드 스타일, PR 프로세스는 [기여 가이드](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)를 참고하세요.

기여자를 위한 빠른 시작 — 리포지토리를 클론하고 `setup-hermes.sh`를 실행해 보세요:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # uv 설치, venv 생성, .[all] 설치, ~/.local/bin/hermes 심볼릭 링크 생성
./hermes              # venv 자동 감지하므로 굳이 source를 먼저 실행할 필요 없음
```

수동 설치 방법 (위와 동일):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## 커뮤니티

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [기술 허브(Skills Hub)](https://agentskills.io)
- 🐛 [이슈(Issues)](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — AT-SPI 접근성 트리, Wayland/X11 입력, 스크린샷, 컴포지터 윈도우 타겟팅을 지원하는 Hermes 및 기타 MCP 호스트용 Linux 데스크톱 제어 MCP 서버.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — 커뮤니티 위챗(WeChat) 브리지: 동일한 위챗 계정에서 Hermes Agent와 OpenClaw를 동시에 실행할 수 있습니다.

---

## 라이선스

MIT — [LICENSE](LICENSE) 파일을 참고하세요.

[Nous Research](https://nousresearch.com)에서 개발했습니다.

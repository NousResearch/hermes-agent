<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
</p>

**[Nous Research](https://nousresearch.com)가 만든 자가 발전형 AI 에이전트.** 학습 루프가 내장된 유일한 에이전트입니다 — 경험으로부터 스킬을 만들고, 사용 중에 개선하며, 지식을 영구 보존하도록 스스로를 자극하고, 과거 대화를 검색하고, 세션을 거치며 사용자에 대한 깊은 모델을 구축합니다. $5짜리 VPS, GPU 클러스터, 또는 유휴 시 거의 비용이 들지 않는 서버리스 인프라에서 실행하세요. 노트북에 묶여있지 않습니다 — Hermes가 클라우드 VM에서 작업하는 동안 Telegram으로 대화할 수 있습니다.

원하는 모든 모델을 사용하세요 — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200개 이상 모델), [NovitaAI](https://novita.ai) (Model API·Agent Sandbox·GPU Cloud를 위한 AI 네이티브 클라우드), [NVIDIA NIM](https://build.nvidia.com) (Nemotron), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI, 또는 사용자 자체 엔드포인트. `hermes model`로 전환 — 코드 변경 없음, 락인 없음.

<table>
<tr><td><b>진짜 터미널 인터페이스</b></td><td>다중 줄 편집, 슬래시 명령 자동 완성, 대화 기록, 끼어들기·리다이렉트, 도구 출력 스트리밍을 갖춘 완전한 TUI.</td></tr>
<tr><td><b>사용자가 있는 곳에 함께</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, CLI — 모두 단일 gateway 프로세스에서. 음성 메모 전사, 플랫폼 간 대화 연속성.</td></tr>
<tr><td><b>닫힌 학습 루프</b></td><td>주기적 자극이 있는 에이전트 큐레이션 메모리. 복잡한 작업 후 자율적 스킬 생성. 사용 중 자가 개선되는 스킬. 세션 간 회상을 위한 LLM 요약 기반 FTS5 세션 검색. <a href="https://github.com/plastic-labs/honcho">Honcho</a> 변증법적 사용자 모델링. <a href="https://agentskills.io">agentskills.io</a> 오픈 표준과 호환.</td></tr>
<tr><td><b>예약 자동화</b></td><td>플랫폼 어디로든 배달 가능한 내장 cron 스케줄러. 일일 리포트, 야간 백업, 주간 감사 — 모두 자연어로, 무인 실행.</td></tr>
<tr><td><b>위임과 병렬화</b></td><td>병렬 작업 흐름을 위한 격리된 서브에이전트 생성. RPC로 도구를 호출하는 Python 스크립트 작성하여, 멀티스텝 파이프라인을 0 컨텍스트 비용 턴으로 압축.</td></tr>
<tr><td><b>노트북뿐 아니라 어디서든 실행</b></td><td>여섯 가지 터미널 백엔드 — local, Docker, SSH, Singularity, Modal, Daytona. Daytona와 Modal은 서버리스 영구화 제공 — 에이전트 환경이 유휴 시 동면하고 요청 시 깨어나, 세션 사이에 거의 비용이 들지 않음. $5짜리 VPS나 GPU 클러스터에서 실행 가능.</td></tr>
<tr><td><b>리서치 준비 완료</b></td><td>다음 세대 도구 호출 모델 훈련용 일괄 trajectory 생성, trajectory 압축.</td></tr>
</table>

---

## 빠른 설치

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (네이티브, PowerShell)

> **참고:** 네이티브 Windows에서는 WSL 없이 Hermes를 실행합니다 — CLI, gateway, TUI, 도구 모두 네이티브로 동작합니다. WSL2를 선호하시면 위의 Linux/macOS 한 줄 명령도 거기서 작동합니다. 버그를 발견하셨다면 [이슈를 등록](https://github.com/NousResearch/hermes-agent/issues)해 주세요.

PowerShell에서 다음을 실행하세요:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

설치 프로그램이 모든 것을 처리합니다: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **그리고 휴대용 Git Bash**(MinGit, `%LOCALAPPDATA%\hermes\git`에 압축 해제 — 관리자 권한 불필요, 시스템 Git 설치와 완전 격리). Hermes는 이 번들 Git Bash를 사용하여 셸 명령을 실행합니다.

이미 Git이 설치되어 있다면 설치 프로그램이 감지하여 그것을 사용합니다. 그렇지 않으면 ~45MB MinGit 다운로드만 필요하며 — 시스템 Git을 건드리거나 간섭하지 않습니다.

> **Android / Termux:** 검증된 수동 경로는 [Termux 가이드](https://hermes-agent.nousresearch.com/docs/getting-started/termux)에 문서화되어 있습니다. Termux에서는 전체 `.[all]` extra가 현재 Android와 호환되지 않는 음성 의존성을 끌어오기 때문에 Hermes가 큐레이팅된 `.[termux]` extra를 설치합니다.
>
> **Windows:** 네이티브 Windows는 완전히 지원됩니다 — 위 PowerShell 한 줄 명령이 모든 것을 설치합니다. WSL2를 선호하시면 Linux 명령도 거기서 동작합니다. 네이티브 Windows 설치는 `%LOCALAPPDATA%\hermes`에 위치하고, WSL2 설치는 Linux와 동일하게 `~/.hermes`에 위치합니다. 현재 WSL2가 특별히 필요한 유일한 Hermes 기능은 브라우저 기반 dashboard 채팅 창입니다(POSIX PTY를 사용 — 클래식 CLI와 gateway는 모두 네이티브로 실행).

설치 후:

```bash
source ~/.bashrc    # 셸 다시 로드 (또는: source ~/.zshrc)
hermes              # 대화 시작!
```

---

## 시작하기

```bash
hermes              # 대화형 CLI — 대화 시작
hermes model        # LLM 공급자와 모델 선택
hermes tools        # 활성화할 도구 구성
hermes config set   # 개별 설정 값 지정
hermes gateway      # 메시징 gateway 시작 (Telegram, Discord 등)
hermes setup        # 전체 설정 마법사 실행 (모든 것을 한 번에 구성)
hermes claw migrate # OpenClaw에서 마이그레이션 (OpenClaw 사용자라면)
hermes update       # 최신 버전으로 업데이트
hermes doctor       # 문제 진단
```

📖 **[전체 문서 →](https://hermes-agent.nousresearch.com/docs/)**

---

## API 키 수집 건너뛰기 — Nous Portal

Hermes는 원하는 어떤 공급자든 사용 가능합니다 — 이건 바뀌지 않습니다. 다만 모델·웹 검색·이미지 생성·TTS·클라우드 브라우저를 위한 다섯 개의 별도 API 키를 수집하고 싶지 않다면, **[Nous Portal](https://portal.nousresearch.com)**이 단일 구독으로 모두 제공합니다:

- **300개 이상 모델** — `/model <name>`으로 아무거나 선택
- **Tool Gateway** — 웹 검색 (Firecrawl), 이미지 생성 (FAL), 음성 합성 (OpenAI), 클라우드 브라우저 (Browser Use) 모두 구독으로 라우팅. 추가 계정 불필요.

새 설치에서 한 명령:

```bash
hermes setup --portal
```

OAuth로 로그인하고, Nous를 공급자로 설정하며, Tool Gateway를 활성화합니다. `hermes portal info`로 언제든 연결 상태 확인 가능. 자세한 내용은 [Tool Gateway 문서 페이지](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) 참조.

원할 때 도구별로 직접 키를 가져와도 됩니다 — gateway는 백엔드 단위이며 전부 아니면 전무가 아닙니다.

---

## CLI vs 메시징 빠른 참조

Hermes에는 두 가지 진입점이 있습니다: `hermes`로 터미널 UI를 시작하거나, gateway를 실행해 Telegram·Discord·Slack·WhatsApp·Signal·이메일로 대화하세요. 대화 중에는 많은 슬래시 명령이 두 인터페이스에서 공유됩니다.

| 동작                              | CLI                                           | 메시징 플랫폼                                                                       |
| --------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------- |
| 대화 시작                          | `hermes`                                      | `hermes gateway setup` + `hermes gateway start` 실행 후 봇에 메시지 전송              |
| 새 대화 시작                       | `/new` 또는 `/reset`                          | `/new` 또는 `/reset`                                                               |
| 모델 변경                          | `/model [provider:model]`                     | `/model [provider:model]`                                                          |
| 페르소나 설정                       | `/personality [name]`                         | `/personality [name]`                                                              |
| 마지막 턴 재시도 또는 취소           | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                  |
| 컨텍스트 압축 / 사용량 확인          | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                          |
| 스킬 찾아보기                       | `/skills` 또는 `/<skill-name>`                | `/<skill-name>`                                                                    |
| 현재 작업 중단                      | `Ctrl+C` 또는 새 메시지 전송                    | `/stop` 또는 새 메시지 전송                                                          |
| 플랫폼별 상태                       | `/platforms`                                  | `/status`, `/sethome`                                                              |

전체 명령 목록은 [CLI 가이드](https://hermes-agent.nousresearch.com/docs/user-guide/cli)와 [메시징 Gateway 가이드](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)를 참조하세요.

---

## 문서

모든 문서는 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**에 있습니다:

| 섹션                                                                                                | 다루는 내용                                                |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)                 | 설치 → 설정 → 첫 대화까지 2분 안에                          |
| [CLI 사용](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                                | 명령, 키 바인딩, 페르소나, 세션                              |
| [구성](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                          | 설정 파일, 공급자, 모델, 모든 옵션                           |
| [메시징 Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)                    | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [보안](https://hermes-agent.nousresearch.com/docs/user-guide/security)                               | 명령 승인, DM 페어링, 컨테이너 격리                          |
| [도구와 도구 세트](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)              | 40개 이상 도구, 도구 세트 시스템, 터미널 백엔드               |
| [스킬 시스템](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)                  | 절차적 메모리, Skills Hub, 스킬 생성                         |
| [메모리](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                      | 영구 메모리, 사용자 프로필, 모범 사례                         |
| [MCP 통합](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                       | 확장 기능을 위한 어떤 MCP 서버든 연결                         |
| [Cron 스케줄링](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)                 | 플랫폼 배달 가능한 예약 작업                                 |
| [컨텍스트 파일](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)         | 모든 대화를 형성하는 프로젝트 컨텍스트                        |
| [아키텍처](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)                  | 프로젝트 구조, 에이전트 루프, 주요 클래스                     |
| [기여](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)                      | 개발 설정, PR 절차, 코드 스타일                              |
| [CLI 참조](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                        | 모든 명령과 플래그                                          |
| [환경 변수](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)              | 전체 환경 변수 참조                                         |

---

## OpenClaw에서 마이그레이션

OpenClaw에서 오신 분이라면 Hermes가 설정·메모리·스킬·API 키를 자동으로 가져올 수 있습니다.

**첫 설정 중:** 설정 마법사(`hermes setup`)가 자동으로 `~/.openclaw`를 감지하고 구성 시작 전에 마이그레이션을 제안합니다.

**설치 후 언제든:**

```bash
hermes claw migrate              # 대화형 마이그레이션 (전체 프리셋)
hermes claw migrate --dry-run    # 마이그레이션 대상 미리보기
hermes claw migrate --preset user-data   # 비밀 정보 없이 마이그레이션
hermes claw migrate --overwrite  # 기존 항목 충돌 시 덮어쓰기
```

가져오는 항목:

- **SOUL.md** — 페르소나 파일
- **메모리** — MEMORY.md와 USER.md 항목
- **스킬** — 사용자가 만든 스킬 → `~/.hermes/skills/openclaw-imports/`
- **명령 허용 목록** — 승인 패턴
- **메시징 설정** — 플랫폼 구성, 허용 사용자, 작업 디렉터리
- **API 키** — 허용된 비밀 (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS 자산** — 작업 공간 오디오 파일
- **작업 공간 지시문** — AGENTS.md (`--workspace-target`과 함께)

모든 옵션은 `hermes claw migrate --help`를 참고하거나, dry-run 미리보기와 함께 대화형 에이전트 안내 마이그레이션을 위해 `openclaw-migration` 스킬을 사용하세요.

---

## 기여

기여를 환영합니다! 개발 설정, 코드 스타일, PR 절차는 [기여 가이드](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)를 참고하세요.

기여자를 위한 빠른 시작 — clone 후 `setup-hermes.sh`로 진행:

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # uv 설치, venv 생성, .[all] 설치, ~/.local/bin/hermes 심링크
./hermes              # venv 자동 감지, 먼저 `source` 할 필요 없음
```

수동 경로 (위와 동등):

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
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [이슈](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Hermes를 비롯한 MCP 호스트용 Linux 데스크톱 제어 MCP 서버. AT-SPI 접근성 트리, Wayland/X11 입력, 스크린샷, 컴포지터 창 타겟팅 지원.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — 커뮤니티 WeChat 브리지: 동일한 WeChat 계정에서 Hermes Agent와 OpenClaw를 함께 실행.

---

## 라이선스

MIT — [LICENSE](LICENSE) 참조.

[Nous Research](https://nousresearch.com)가 만들었습니다.

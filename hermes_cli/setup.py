"""
Hermes Agent 설정 마법사.

모듈식 마법사이며 각 섹션을 독립적으로 실행할 수 있습니다:
  1. 모델 및 Provider — AI provider와 모델 선택
  2. 터미널 백엔드 — 에이전트가 명령을 실행할 위치
  3. 에이전트 설정 — 반복 수, 압축, 세션 리셋
  4. 메시징 플랫폼 — Telegram, Discord 등 연결
  5. 도구 — TTS, 웹 검색, 이미지 생성 등 설정

설정 파일은 쉽게 접근할 수 있도록 ~/.hermes/ 아래에 저장됩니다.
"""

import importlib.util
import logging
import os
import shutil
import sys
import copy
from pathlib import Path
from typing import Optional, Dict, Any

from hermes_cli.nous_subscription import (
    apply_nous_provider_defaults,
    get_nous_subscription_features,
)
from tools.tool_backend_helpers import managed_nous_tools_enabled
from hermes_constants import get_optional_skills_dir

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_DOCS_BASE = "https://hermes-agent.nousresearch.com/docs"


def _model_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    current_model = config.get("model")
    if isinstance(current_model, dict):
        return dict(current_model)
    if isinstance(current_model, str) and current_model.strip():
        return {"default": current_model.strip()}
    return {}


def _get_credential_pool_strategies(config: Dict[str, Any]) -> Dict[str, str]:
    strategies = config.get("credential_pool_strategies")
    return dict(strategies) if isinstance(strategies, dict) else {}


def _set_credential_pool_strategy(config: Dict[str, Any], provider: str, strategy: str) -> None:
    if not provider:
        return
    strategies = _get_credential_pool_strategies(config)
    strategies[provider] = strategy
    config["credential_pool_strategies"] = strategies


def _supports_same_provider_pool_setup(provider: str) -> bool:
    if not provider or provider == "custom":
        return False
    if provider == "openrouter":
        return True
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig:
        return False
    return pconfig.auth_type in {"api_key", "oauth_device_code"}


# Default model lists per provider — used as fallback when the live
# /models endpoint can't be reached.
_DEFAULT_PROVIDER_MODELS = {
    "copilot-acp": [
        "copilot-acp",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-opus-4.6",
        "claude-sonnet-4.6",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-2.5-pro",
        "grok-code-fast-1",
    ],
    "gemini": [
        "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        "gemma-4-31b-it", "gemma-4-26b-it",
    ],
    "zai": ["glm-5.1", "glm-5", "glm-4.7", "glm-4.5", "glm-4.5-flash"],
    "kimi-coding": ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview"],
    "kimi-coding-cn": ["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-turbo-preview"],
    "arcee": ["trinity-large-thinking", "trinity-large-preview", "trinity-mini"],
    "minimax": ["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"],
    "minimax-cn": ["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"],
    "ai-gateway": ["anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5", "google/gemini-3-flash"],
    "kilocode": ["anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "google/gemini-3-pro-preview", "google/gemini-3-flash-preview"],
    "opencode-zen": ["gpt-5.4", "gpt-5.3-codex", "claude-sonnet-4-6", "gemini-3-flash", "glm-5", "kimi-k2.5", "minimax-m2.7"],
    "opencode-go": ["glm-5", "kimi-k2.5", "mimo-v2-pro", "mimo-v2-omni", "minimax-m2.5", "minimax-m2.7"],
    "huggingface": [
        "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct", "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-V3.2", "moonshotai/Kimi-K2.5",
    ],
}


def _current_reasoning_effort(config: Dict[str, Any]) -> str:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    return ""


def _set_reasoning_effort(config: Dict[str, Any], effort: str) -> None:
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        config["agent"] = agent_cfg
    agent_cfg["reasoning_effort"] = effort




# Import config helpers
from hermes_cli.config import (
    DEFAULT_CONFIG,
    get_hermes_home,
    get_config_path,
    get_env_path,
    load_config,
    save_config,
    save_env_value,
    get_env_value,
    ensure_hermes_home,
)
# display_hermes_home imported lazily at call sites (stale-module safety during hermes update)

from hermes_cli.colors import Colors, color


def print_header(title: str):
    """Print a section header."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


from hermes_cli.cli_output import (  # noqa: E402
    print_error,
    print_info,
    print_success,
    print_warning,
)


def is_interactive_stdin() -> bool:
    """Return True when stdin looks like a usable interactive TTY."""
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return False
    try:
        return bool(stdin.isatty())
    except Exception:
        return False


def print_noninteractive_setup_guidance(reason: str | None = None) -> None:
    """Print guidance for headless/non-interactive setup flows."""
    print()
    print(color("⚕ Hermes 설정 — 비대화형 모드", Colors.CYAN, Colors.BOLD))
    print()
    if reason:
        print_info(reason)
    print_info("대화형 마법사는 여기서 사용할 수 없습니다.")
    print()
    print_info("환경 변수 또는 config 명령으로 Hermes를 설정하세요:")
    print_info("  hermes config set model.provider custom")
    print_info("  hermes config set model.base_url http://localhost:8080/v1")
    print_info("  hermes config set model.default your-model-name")
    print()
    print_info("또는 환경 변수에 OPENROUTER_API_KEY / OPENAI_API_KEY를 설정하세요.")
    print_info("전체 마법사를 사용하려면 대화형 터미널에서 'hermes setup'을 실행하세요.")
    print()


def prompt(question: str, default: str = None, password: bool = False) -> str:
    """Prompt for input with optional default."""
    if default:
        display = f"{question} [{default}]: "
    else:
        display = f"{question}: "

    try:
        if password:
            import getpass

            value = getpass.getpass(color(display, Colors.YELLOW))
        else:
            value = input(color(display, Colors.YELLOW))

        return value.strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


def _curses_prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu using curses. Delegates to curses_radiolist."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=-1)



def prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Prompt for a choice from a list with arrow key navigation.

    Escape keeps the current default (skips the question).
    Ctrl+C exits the wizard.
    """
    idx = _curses_prompt_choice(question, choices, default)
    if idx >= 0:
        if idx == default:
            print_info("  건너뜀 (현재 설정 유지)")
            print()
            return default
        print()
        return idx

    print(color(question, Colors.YELLOW))
    for i, choice in enumerate(choices):
        marker = "●" if i == default else "○"
        if i == default:
            print(color(f"  {marker} {choice}", Colors.GREEN))
        else:
            print(f"  {marker} {choice}")

    print_info(f"  기본값은 Enter ({default + 1})  종료는 Ctrl+C")

    while True:
        try:
            value = input(
                color(f"  선택 [1-{len(choices)}] ({default + 1}): ", Colors.DIM)
            )
            if not value:
                return default
            idx = int(value) - 1
            if 0 <= idx < len(choices):
                return idx
            print_error(f"1부터 {len(choices)} 사이의 숫자를 입력하세요")
        except ValueError:
            print_error("숫자를 입력하세요")
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no. Ctrl+C exits, empty input returns default."""
    default_str = "Y/n" if default else "y/N"

    while True:
        try:
            value = (
                input(color(f"{question} [{default_str}]: ", Colors.YELLOW))
                .strip()
                .lower()
            )
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)

        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print_error("'y' 또는 'n'을 입력하세요")


def prompt_checklist(title: str, items: list, pre_selected: list = None) -> list:
    """
    Display a multi-select checklist and return the indices of selected items.

    Each item in `items` is a display string. `pre_selected` is a list of
    indices that should be checked by default. A "Continue →" option is
    appended at the end — the user toggles items with Space and confirms
    with Enter on "Continue →".

    Falls back to a numbered toggle interface when simple_term_menu is
    unavailable.

    Returns:
        List of selected indices (not including the Continue option).
    """
    if pre_selected is None:
        pre_selected = []

    from hermes_cli.curses_ui import curses_checklist

    chosen = curses_checklist(
        title,
        items,
        set(pre_selected),
        cancel_returns=set(pre_selected),
    )
    return sorted(chosen)


def _prompt_api_key(var: dict):
    """Display a nicely formatted API key input screen for a single env var."""
    tools = var.get("tools", [])
    tools_str = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_str += f", +{len(tools) - 3} more"

    print()
    print(color(f"  ─── {var.get('description', var['name'])} ───", Colors.CYAN))
    print()
    if tools_str:
        print_info(f"  활성화 기능: {tools_str}")
    if var.get("url"):
        print_info(f"  키 발급 위치: {var['url']}")
    print()

    if var.get("password"):
        value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
    else:
        value = prompt(f"  {var.get('prompt', var['name'])}")

    if value:
        save_env_value(var["name"], value)
        print_success("  ✓ 저장 완료")
    else:
        print_warning("  건너뜀 (나중에 'hermes setup'으로 설정 가능)")


def _print_setup_summary(config: dict, hermes_home):
    """Print the setup completion summary."""
    # Tool availability summary
    print()
    print_header("도구 사용 가능 요약")

    tool_status = []
    subscription_features = get_nous_subscription_features(config)

    # Vision — use the same runtime resolver as the actual vision tools
    try:
        from agent.auxiliary_client import get_available_vision_backends

        _vision_backends = get_available_vision_backends()
    except Exception:
        _vision_backends = []

    if _vision_backends:
        tool_status.append(("비전 (이미지 분석)", True, None))
    else:
        tool_status.append(("비전 (이미지 분석)", False, "'hermes setup'을 실행해 설정"))

    # Mixture of Agents — requires OpenRouter specifically (calls multiple models)
    if get_env_value("OPENROUTER_API_KEY"):
        tool_status.append(("에이전트 혼합", True, None))
    else:
        tool_status.append(("에이전트 혼합", False, "OPENROUTER_API_KEY"))

    # Web tools (Exa, Parallel, Firecrawl, or Tavily)
    if subscription_features.web.managed_by_nous:
        tool_status.append(("웹 검색 및 추출 (Nous 구독)", True, None))
    elif subscription_features.web.available:
        label = "웹 검색 및 추출"
        if subscription_features.web.current_provider:
            label = f"웹 검색 및 추출 ({subscription_features.web.current_provider})"
        tool_status.append((label, True, None))
    else:
        tool_status.append(("웹 검색 및 추출", False, "EXA_API_KEY, PARALLEL_API_KEY, FIRECRAWL_API_KEY/FIRECRAWL_API_URL 또는 TAVILY_API_KEY"))

    # Browser tools (local Chromium, Camofox, Browserbase, Browser Use, or Firecrawl)
    browser_provider = subscription_features.browser.current_provider
    if subscription_features.browser.managed_by_nous:
        tool_status.append(("브라우저 자동화 (Nous Browser Use)", True, None))
    elif subscription_features.browser.available:
        label = "브라우저 자동화"
        if browser_provider:
            label = f"브라우저 자동화 ({browser_provider})"
        tool_status.append((label, True, None))
    else:
        missing_browser_hint = "npm install -g agent-browser 실행 후 CAMOFOX_URL을 설정하거나 Browser Use/Browserbase를 구성하세요"
        if browser_provider == "Browserbase":
            missing_browser_hint = (
                "npm install -g agent-browser 실행 후 "
                "BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID를 설정하세요"
            )
        elif browser_provider == "Browser Use":
            missing_browser_hint = (
                "npm install -g agent-browser 실행 후 BROWSER_USE_API_KEY를 설정하세요"
            )
        elif browser_provider == "Camofox":
            missing_browser_hint = "CAMOFOX_URL"
        elif browser_provider == "Local browser":
            missing_browser_hint = "npm install -g agent-browser"
        tool_status.append(
            ("브라우저 자동화", False, missing_browser_hint)
        )

    # FAL (image generation)
    if subscription_features.image_gen.managed_by_nous:
        tool_status.append(("이미지 생성 (Nous 구독)", True, None))
    elif subscription_features.image_gen.available:
        tool_status.append(("이미지 생성", True, None))
    else:
        tool_status.append(("이미지 생성", False, "FAL_KEY"))

    # TTS — show configured provider
    tts_provider = config.get("tts", {}).get("provider", "edge")
    if subscription_features.tts.managed_by_nous:
        tool_status.append(("텍스트 음성 변환 (Nous 구독의 OpenAI)", True, None))
    elif tts_provider == "elevenlabs" and get_env_value("ELEVENLABS_API_KEY"):
        tool_status.append(("텍스트 음성 변환 (ElevenLabs)", True, None))
    elif tts_provider == "openai" and (
        get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY")
    ):
        tool_status.append(("텍스트 음성 변환 (OpenAI)", True, None))
    elif tts_provider == "minimax" and get_env_value("MINIMAX_API_KEY"):
        tool_status.append(("텍스트 음성 변환 (MiniMax)", True, None))
    elif tts_provider == "mistral" and get_env_value("MISTRAL_API_KEY"):
        tool_status.append(("텍스트 음성 변환 (Mistral Voxtral)", True, None))
    elif tts_provider == "neutts":
        try:
            import importlib.util
            neutts_ok = importlib.util.find_spec("neutts") is not None
        except Exception:
            neutts_ok = False
        if neutts_ok:
            tool_status.append(("텍스트 음성 변환 (NeuTTS 로컬)", True, None))
        else:
            tool_status.append(("텍스트 음성 변환 (NeuTTS — 설치되지 않음)", False, "'hermes setup tts' 실행"))
    else:
        tool_status.append(("텍스트 음성 변환 (Edge TTS)", True, None))

    if subscription_features.modal.managed_by_nous:
        tool_status.append(("Modal 실행 (Nous 구독)", True, None))
    elif config.get("terminal", {}).get("backend") == "modal":
        if subscription_features.modal.direct_override:
            tool_status.append(("Modal 실행 (직접 Modal)", True, None))
        else:
            tool_status.append(("Modal 실행", False, "'hermes setup terminal' 실행"))
    elif managed_nous_tools_enabled() and subscription_features.nous_auth_present:
        tool_status.append(("Modal 실행 (Nous 구독으로 선택 사용 가능)", True, None))

    # Tinker + WandB (RL training)
    if get_env_value("TINKER_API_KEY") and get_env_value("WANDB_API_KEY"):
        tool_status.append(("RL 학습 (Tinker)", True, None))
    elif get_env_value("TINKER_API_KEY"):
        tool_status.append(("RL 학습 (Tinker)", False, "WANDB_API_KEY"))
    else:
        tool_status.append(("RL 학습 (Tinker)", False, "TINKER_API_KEY"))

    # Home Assistant
    if get_env_value("HASS_TOKEN"):
        tool_status.append(("스마트 홈 (Home Assistant)", True, None))

    # Skills Hub
    if get_env_value("GITHUB_TOKEN"):
        tool_status.append(("스킬 허브 (GitHub)", True, None))
    else:
        tool_status.append(("스킬 허브 (GitHub)", False, "GITHUB_TOKEN"))

    # Terminal (always available if system deps met)
    tool_status.append(("터미널/명령어", True, None))

    # Task planning (always available, in-memory)
    tool_status.append(("작업 계획(todo)", True, None))

    # Skills (always available -- bundled skills + user-created skills)
    tool_status.append(("스킬 (보기, 생성, 편집)", True, None))

    # Print status
    available_count = sum(1 for _, avail, _ in tool_status if avail)
    total_count = len(tool_status)

    print_info(f"{available_count}/{total_count}개 도구 범주 사용 가능:")
    print()

    for name, available, missing_var in tool_status:
        if available:
            print(f"   {color('✓', Colors.GREEN)} {name}")
        else:
            print(
                f"   {color('✗', Colors.RED)} {name} {color(f'(누락 - {missing_var})', Colors.DIM)}"
            )

    print()

    disabled_tools = [(name, var) for name, avail, var in tool_status if not avail]
    if disabled_tools:
        print_warning(
            "일부 도구가 비활성화되어 있습니다. 'hermes setup tools'를 실행해 설정하거나,"
        )
        from hermes_constants import display_hermes_home as _dhh
        print_warning(f"{_dhh()}/.env를 직접 수정해 누락된 API 키를 추가하세요.")
        print()

    # Done banner
    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐", Colors.GREEN
        )
    )
    print(
        color(
            "│            ✓ 설정이 완료되었습니다!                   │", Colors.GREEN
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘", Colors.GREEN
        )
    )
    print()

    # Show file locations prominently
    from hermes_constants import display_hermes_home as _dhh
    print(color(f"📁 모든 파일은 {_dhh()}/ 아래에 있습니다:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('설정:', Colors.YELLOW)}     {get_config_path()}")
    print(f"   {color('API 키:', Colors.YELLOW)}   {get_env_path()}")
    print(
        f"   {color('데이터:', Colors.YELLOW)}   {hermes_home}/cron/, sessions/, logs/"
    )
    print()

    print(color("─" * 60, Colors.DIM))
    print()
    print(color("📝 설정을 수정하려면:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('hermes setup', Colors.GREEN)}             전체 마법사 다시 실행")
    print(f"   {color('hermes setup model', Colors.GREEN)}       모델/provider 변경")
    print(f"   {color('hermes setup terminal', Colors.GREEN)}    터미널 백엔드 변경")
    print(f"   {color('hermes setup gateway', Colors.GREEN)}     메시징 설정")
    print(f"   {color('hermes setup tools', Colors.GREEN)}       도구 provider 설정")
    print()

    print(f"   {color('hermes config', Colors.GREEN)}            현재 설정 보기")
    print(
        f"   {color('hermes config edit', Colors.GREEN)}       편집기에서 config 열기"
    )
    print(f"   {color('hermes config set <key> <value>', Colors.GREEN)}")
    print("                           특정 값을 직접 설정")
    print()
    print("   또는 파일을 직접 수정하세요:")
    print(f"   {color(f'nano {get_config_path()}', Colors.DIM)}")
    print(f"   {color(f'nano {get_env_path()}', Colors.DIM)}")
    print()

    print(color("─" * 60, Colors.DIM))
    print()
    print(color("🚀 이제 사용할 준비가 되었습니다!", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('hermes', Colors.GREEN)}              대화 시작")
    print(f"   {color('hermes gateway', Colors.GREEN)}      메시징 gateway 시작")
    print(f"   {color('hermes doctor', Colors.GREEN)}       문제 점검")
    print()

def _prompt_container_resources(config: dict):
    """Prompt for container resource settings (Docker, Singularity, Modal, Daytona)."""
    terminal = config.setdefault("terminal", {})

    print()
    print_info("컨테이너 리소스 설정:")

    # Persistence
    current_persist = terminal.get("container_persistent", True)
    persist_label = "yes" if current_persist else "no"
    print_info("  영구 파일시스템을 켜면 세션 사이에도 파일이 유지됩니다.")
    print_info("  매번 초기화되는 임시 샌드박스를 원하면 'no'로 설정하세요.")
    persist_str = prompt(
        "  세션 간 파일시스템을 유지할까요? (yes/no)", persist_label
    )
    terminal["container_persistent"] = persist_str.lower() in ("yes", "true", "y", "1")

    # CPU
    current_cpu = terminal.get("container_cpu", 1)
    cpu_str = prompt("  CPU cores", str(current_cpu))
    try:
        terminal["container_cpu"] = float(cpu_str)
    except ValueError:
        pass

    # Memory
    current_mem = terminal.get("container_memory", 5120)
    mem_str = prompt("  Memory in MB (5120 = 5GB)", str(current_mem))
    try:
        terminal["container_memory"] = int(mem_str)
    except ValueError:
        pass

    # Disk
    current_disk = terminal.get("container_disk", 51200)
    disk_str = prompt("  Disk in MB (51200 = 50GB)", str(current_disk))
    try:
        terminal["container_disk"] = int(disk_str)
    except ValueError:
        pass


# Tool categories and provider config are now in tools_config.py (shared
# between `hermes tools` and `hermes setup tools`).


# =============================================================================
# Section 1: Model & Provider Configuration
# =============================================================================



def setup_model_provider(config: dict, *, quick: bool = False):
    """Configure the inference provider and default model.

    Delegates to ``cmd_model()`` (the same flow used by ``hermes model``)
    for provider selection, credential prompting, and model picking.
    This ensures a single code path for all provider setup — any new
    provider added to ``hermes model`` is automatically available here.

    When *quick* is True, skips credential rotation, vision, and TTS
    configuration — used by the streamlined first-time quick setup.
    """
    from hermes_cli.config import load_config, save_config

    print_header("모델 및 Provider")
    print_info("주요 채팅 모델에 연결하는 방법을 선택하세요.")
    print_info(f"   Guide: {_DOCS_BASE}/integrations/providers")
    print()

    # Delegate to the shared hermes model flow — handles provider picker,
    # credential prompting, model selection, and config persistence.
    from hermes_cli.main import select_provider_and_model
    try:
        select_provider_and_model()
    except (SystemExit, KeyboardInterrupt):
        print()
        print_info("Provider 설정을 건너뛰었습니다.")
    except Exception as exc:
        logger.debug("select_provider_and_model error during setup: %s", exc)
        print_warning(f"Provider 설정 중 오류가 발생했습니다: {exc}")
        print_info("나중에 'hermes model'로 다시 시도할 수 있습니다")

    # Re-sync the wizard's config dict from what cmd_model saved to disk.
    # This is critical: cmd_model writes to disk via its own load/save cycle,
    # and the wizard's final save_config(config) must not overwrite those
    # changes with stale values (#4172).
    _refreshed = load_config()
    config["model"] = _refreshed.get("model", config.get("model"))
    if "custom_providers" in _refreshed:
        config["custom_providers"] = _refreshed["custom_providers"]
    else:
        config.pop("custom_providers", None)

    # Derive the selected provider for downstream steps (vision setup).
    selected_provider = None
    _m = config.get("model")
    if isinstance(_m, dict):
        selected_provider = _m.get("provider")

    nous_subscription_selected = selected_provider == "nous"

    # ── Same-provider fallback & rotation setup (full setup only) ──
    if not quick and _supports_same_provider_pool_setup(selected_provider):
        try:
            from types import SimpleNamespace
            from agent.credential_pool import load_pool
            from hermes_cli.auth_commands import auth_add_command

            pool = load_pool(selected_provider)
            entries = pool.entries()
            entry_count = len(entries)
            manual_count = sum(1 for entry in entries if str(getattr(entry, "source", "")).startswith("manual"))
            auto_count = entry_count - manual_count
            print()
            print_header("동일 Provider fallback 및 순환")
            print_info(
                "Hermes는 하나의 provider에 여러 자격 증명을 보관하고, 자격 증명이 소진되거나"
            )
            print_info(
                "속도 제한에 걸리면 다른 자격 증명으로 순환할 수 있습니다."
            )
            print_info(
                "이렇게 하면 기본 provider는 유지하면서 quota 문제로 인한 중단을 줄일 수 있습니다."
            )
            print()
            if auto_count > 0:
                print_info(
                    f"현재 {selected_provider} 풀 자격 증명: {entry_count}개 "
                    f"(수동 {manual_count}, env/공유 인증에서 자동 감지 {auto_count})"
                )
            else:
                print_info(f"현재 {selected_provider} 풀 자격 증명: {entry_count}개")

            while prompt_yes_no("같은 provider fallback용 자격 증명을 하나 더 추가할까요?", False):
                auth_add_command(
                    SimpleNamespace(
                        provider=selected_provider,
                        auth_type="",
                        label=None,
                        api_key=None,
                        portal_url=None,
                        inference_url=None,
                        client_id=None,
                        scope=None,
                        no_browser=False,
                        timeout=15.0,
                        insecure=False,
                        ca_bundle=None,
                        min_key_ttl_seconds=5 * 60,
                    )
                )
                pool = load_pool(selected_provider)
                entry_count = len(pool.entries())
                print_info(f"Provider 풀에 자격 증명이 이제 {entry_count}개 있습니다.")

            if entry_count > 1:
                strategy_labels = [
                    "fill-first / sticky — 첫 번째 건강한 자격 증명을 소진될 때까지 계속 사용",
                    "round robin — 선택할 때마다 다음 건강한 자격 증명으로 순환",
                    "random — 매번 건강한 자격 증명 중 하나를 무작위 선택",
                ]
                current_strategy = _get_credential_pool_strategies(config).get(selected_provider, "fill_first")
                default_strategy_idx = {
                    "fill_first": 0,
                    "round_robin": 1,
                    "random": 2,
                }.get(current_strategy, 0)
                strategy_idx = prompt_choice(
                    "동일 provider 순환 전략을 선택하세요:",
                    strategy_labels,
                    default_strategy_idx,
                )
                strategy_value = ["fill_first", "round_robin", "random"][strategy_idx]
                _set_credential_pool_strategy(config, selected_provider, strategy_value)
                print_success(f"{selected_provider} 순환 전략 저장 완료: {strategy_value}")
        except Exception as exc:
            logger.debug("Could not configure same-provider fallback in setup: %s", exc)

    # ── Vision & Image Analysis Setup (full setup only) ──
    if quick:
        _vision_needs_setup = False
    else:
        try:
            from agent.auxiliary_client import get_available_vision_backends
            _vision_backends = set(get_available_vision_backends())
        except Exception:
            _vision_backends = set()

        _vision_needs_setup = not bool(_vision_backends)

        if selected_provider in _vision_backends:
            _vision_needs_setup = False

    if _vision_needs_setup:
        _prov_names = {
            "nous-api": "Nous Portal API key",
            "copilot": "GitHub Copilot",
            "copilot-acp": "GitHub Copilot ACP",
            "zai": "Z.AI / GLM",
            "kimi-coding": "Kimi / Moonshot",
            "kimi-coding-cn": "Kimi / Moonshot (China)",
            "minimax": "MiniMax",
            "minimax-cn": "MiniMax CN",
            "anthropic": "Anthropic",
            "ai-gateway": "Vercel AI Gateway",
            "custom": "your custom endpoint",
        }
        _prov_display = _prov_names.get(selected_provider, selected_provider or "your provider")

        print()
        print_header("비전 및 이미지 분석 (선택 사항)")
        print_info(f"Vision은 별도의 멀티모달 백엔드를 사용합니다. {_prov_display}")
        print_info("현재는 Hermes가 vision용으로 자동 활용할 수 있는 백엔드를 제공하지 않으므로,")
        print_info("지금 백엔드를 선택하거나 건너뛰고 나중에 설정하세요.")
        print()

        _vision_choices = [
            "OpenRouter — Gemini 사용 (openrouter.ai/keys의 무료 티어 가능)",
            "OpenAI 호환 엔드포인트 — base URL, API key, vision 모델 지정",
            "지금은 건너뛰기",
        ]
        _vision_idx = prompt_choice("vision 설정:", _vision_choices, 2)

        if _vision_idx == 0:  # OpenRouter
            _or_key = prompt("  OpenRouter API 키", password=True).strip()
            if _or_key:
                save_env_value("OPENROUTER_API_KEY", _or_key)
                print_success("OpenRouter 키 저장 완료 — vision은 Gemini를 사용합니다")
            else:
                print_info("건너뜀 — vision을 사용할 수 없습니다")
        elif _vision_idx == 1:  # OpenAI-compatible endpoint
            _base_url = prompt("  Base URL (비워두면 OpenAI)").strip() or "https://api.openai.com/v1"
            _api_key_label = "  API 키"
            if "api.openai.com" in _base_url.lower():
                _api_key_label = "  OpenAI API 키"
            _oai_key = prompt(_api_key_label, password=True).strip()
            if _oai_key:
                save_env_value("OPENAI_API_KEY", _oai_key)
                # Save vision base URL to config (not .env — only secrets go there)
                _vaux = config.setdefault("auxiliary", {}).setdefault("vision", {})
                _vaux["base_url"] = _base_url
                if "api.openai.com" in _base_url.lower():
                    _oai_vision_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
                    _vm_choices = _oai_vision_models + ["기본값 사용 (gpt-4o-mini)"]
                    _vm_idx = prompt_choice("vision 모델 선택:", _vm_choices, 0)
                    _selected_vision_model = (
                        _oai_vision_models[_vm_idx]
                        if _vm_idx < len(_oai_vision_models)
                        else "gpt-4o-mini"
                    )
                else:
                    _selected_vision_model = prompt("  Vision 모델 (비워두면 메인/커스텀 기본값 사용)").strip()
                save_env_value("AUXILIARY_VISION_MODEL", _selected_vision_model)
                print_success(
                    f"Vision 설정 완료: {_base_url}"
                    + (f" ({_selected_vision_model})" if _selected_vision_model else "")
                )
            else:
                print_info("건너뜀 — vision을 사용할 수 없습니다")
        else:
            print_info("건너뜀 — 나중에 'hermes setup'으로 추가하거나 AUXILIARY_VISION_* 설정을 구성하세요")


    if selected_provider == "nous" and nous_subscription_selected:
        changed_defaults = apply_nous_provider_defaults(config)
        current_tts = str(config.get("tts", {}).get("provider") or "edge")
        if "tts" in changed_defaults:
            print_success("TTS provider를 OpenAI TTS (Nous 구독)로 설정했습니다")
        else:
            print_info(f"기존 TTS provider를 유지합니다: {current_tts}")

    save_config(config)

    if not quick and selected_provider != "nous":
        _setup_tts_provider(config)


# =============================================================================
# Section 1b: TTS Provider Configuration
# =============================================================================


def _check_espeak_ng() -> bool:
    """Check if espeak-ng is installed."""
    import shutil
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _install_neutts_deps() -> bool:
    """Install NeuTTS dependencies with user approval. Returns True on success."""
    import subprocess
    import sys

    # Check espeak-ng
    if not _check_espeak_ng():
        print()
        print_warning("NeuTTS requires espeak-ng for phonemization.")
        if sys.platform == "darwin":
            print_info("Install with: brew install espeak-ng")
        elif sys.platform == "win32":
            print_info("Install with: choco install espeak-ng")
        else:
            print_info("Install with: sudo apt install espeak-ng")
        print()
        if prompt_yes_no("Install espeak-ng now?", True):
            try:
                if sys.platform == "darwin":
                    subprocess.run(["brew", "install", "espeak-ng"], check=True)
                elif sys.platform == "win32":
                    subprocess.run(["choco", "install", "espeak-ng", "-y"], check=True)
                else:
                    subprocess.run(["sudo", "apt", "install", "-y", "espeak-ng"], check=True)
                print_success("espeak-ng installed")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print_warning(f"Could not install espeak-ng automatically: {e}")
                print_info("Please install it manually and re-run setup.")
                return False
        else:
            print_warning("espeak-ng is required for NeuTTS. Install it manually before using NeuTTS.")

    # Install neutts Python package
    print()
    print_info("Installing neutts Python package...")
    print_info("This will also download the TTS model (~300MB) on first use.")
    print()
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "neutts[all]", "--quiet"],
            check=True, timeout=300,
        )
        print_success("neutts installed successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print_error(f"Failed to install neutts: {e}")
        print_info("Try manually: python -m pip install -U neutts[all]")
        return False


def _setup_tts_provider(config: dict):
    """Interactive TTS provider selection with install flow for NeuTTS."""
    tts_config = config.get("tts", {})
    current_provider = tts_config.get("provider", "edge")
    subscription_features = get_nous_subscription_features(config)

    provider_labels = {
        "edge": "Edge TTS",
        "elevenlabs": "ElevenLabs",
        "openai": "OpenAI TTS",
        "minimax": "MiniMax TTS",
        "mistral": "Mistral Voxtral TTS",
        "neutts": "NeuTTS",
    }
    current_label = provider_labels.get(current_provider, current_provider)

    print()
    print_header("텍스트 음성 변환 provider (선택 사항)")
    print_info(f"현재 설정: {current_label}")
    print()

    choices = []
    providers = []
    if managed_nous_tools_enabled() and subscription_features.nous_auth_present:
        choices.append("Nous Subscription (managed OpenAI TTS, billed to your subscription)")
        providers.append("nous-openai")
    choices.extend(
        [
            "Edge TTS (free, cloud-based, no setup needed)",
            "ElevenLabs (premium quality, needs API key)",
            "OpenAI TTS (good quality, needs API key)",
            "MiniMax TTS (high quality with voice cloning, needs API key)",
            "Mistral Voxtral TTS (multilingual, native Opus, needs API key)",
            "NeuTTS (local on-device, free, ~300MB model download)",
        ]
    )
    providers.extend(["edge", "elevenlabs", "openai", "minimax", "mistral", "neutts"])
    choices.append(f"현재 설정 유지 ({current_label})")
    keep_current_idx = len(choices) - 1
    idx = prompt_choice("TTS provider를 선택하세요:", choices, keep_current_idx)

    if idx == keep_current_idx:
        return

    selected = providers[idx]
    selected_via_nous = selected == "nous-openai"
    if selected == "nous-openai":
        selected = "openai"
        print_info("OpenAI TTS will use the managed Nous gateway and bill to your subscription.")
        if get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY"):
            print_warning(
                "Direct OpenAI credentials are still configured and may take precedence until removed from ~/.hermes/.env."
            )

    if selected == "neutts":
        # Check if already installed
        try:
            import importlib.util
            already_installed = importlib.util.find_spec("neutts") is not None
        except Exception:
            already_installed = False

        if already_installed:
            print_success("NeuTTS is already installed")
        else:
            print()
            print_info("NeuTTS requires:")
            print_info("  • Python package: neutts (~50MB install + ~300MB model on first use)")
            print_info("  • System package: espeak-ng (phonemizer)")
            print()
            if prompt_yes_no("Install NeuTTS dependencies now?", True):
                if not _install_neutts_deps():
                    print_warning("NeuTTS installation incomplete. Falling back to Edge TTS.")
                    selected = "edge"
            else:
                print_info("Skipping install. Set tts.provider to 'neutts' after installing manually.")
                selected = "edge"

    elif selected == "elevenlabs":
        existing = get_env_value("ELEVENLABS_API_KEY")
        if not existing:
            print()
            api_key = prompt("ElevenLabs API key", password=True)
            if api_key:
                save_env_value("ELEVENLABS_API_KEY", api_key)
                print_success("ElevenLabs API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "openai" and not selected_via_nous:
        existing = get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY")
        if not existing:
            print()
            api_key = prompt("OpenAI API key for TTS", password=True)
            if api_key:
                save_env_value("VOICE_TOOLS_OPENAI_KEY", api_key)
                print_success("OpenAI TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "minimax":
        existing = get_env_value("MINIMAX_API_KEY")
        if not existing:
            print()
            api_key = prompt("MiniMax API key for TTS", password=True)
            if api_key:
                save_env_value("MINIMAX_API_KEY", api_key)
                print_success("MiniMax TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    elif selected == "mistral":
        existing = get_env_value("MISTRAL_API_KEY")
        if not existing:
            print()
            api_key = prompt("Mistral API key for TTS", password=True)
            if api_key:
                save_env_value("MISTRAL_API_KEY", api_key)
                print_success("Mistral TTS API key saved")
            else:
                print_warning("No API key provided. Falling back to Edge TTS.")
                selected = "edge"

    # Save the selection
    if "tts" not in config:
        config["tts"] = {}
    config["tts"]["provider"] = selected
    save_config(config)
    print_success(f"TTS provider set to: {provider_labels.get(selected, selected)}")


def setup_tts(config: dict):
    """Standalone TTS setup (for 'hermes setup tts')."""
    _setup_tts_provider(config)


# =============================================================================
# Section 2: Terminal Backend Configuration
# =============================================================================


def setup_terminal_backend(config: dict):
    """Configure the terminal execution backend."""
    import platform as _platform
    import shutil

    print_header("터미널 백엔드")
    print_info("Hermes가 셸 명령과 코드를 어디서 실행할지 선택하세요.")
    print_info("이 설정은 도구 실행, 파일 접근, 격리에 영향을 줍니다.")
    print_info(f"   Guide: {_DOCS_BASE}/developer-guide/environments")
    print()

    current_backend = config.get("terminal", {}).get("backend", "local")
    is_linux = _platform.system() == "Linux"

    # Build backend choices with descriptions
    terminal_choices = [
        "Local - run directly on this machine (default)",
        "Docker - isolated container with configurable resources",
        "Modal - serverless cloud sandbox",
        "SSH - run on a remote machine",
        "Daytona - persistent cloud development environment",
    ]
    idx_to_backend = {0: "local", 1: "docker", 2: "modal", 3: "ssh", 4: "daytona"}
    backend_to_idx = {"local": 0, "docker": 1, "modal": 2, "ssh": 3, "daytona": 4}

    next_idx = 5
    if is_linux:
        terminal_choices.append("Singularity/Apptainer - HPC-friendly container")
        idx_to_backend[next_idx] = "singularity"
        backend_to_idx["singularity"] = next_idx
        next_idx += 1

    # Add keep current option
    keep_current_idx = next_idx
    terminal_choices.append(f"현재 설정 유지 ({current_backend})")
    idx_to_backend[keep_current_idx] = current_backend

    terminal_idx = prompt_choice(
        "터미널 백엔드를 선택하세요:", terminal_choices, keep_current_idx
    )

    selected_backend = idx_to_backend.get(terminal_idx)

    if terminal_idx == keep_current_idx:
        print_info(f"현재 백엔드 유지: {current_backend}")
        return

    config.setdefault("terminal", {})["backend"] = selected_backend

    if selected_backend == "local":
        print_success("터미널 백엔드: Local")
        print_info("명령어를 현재 머신에서 직접 실행합니다.")

        # CWD for messaging
        print()
        print_info("메시징 세션의 작업 디렉터리:")
        print_info("  Telegram/Discord에서 Hermes를 사용할 때")
        print_info(
            "  에이전트가 여기서 시작합니다. CLI 모드는 항상 현재 디렉터리에서 시작합니다."
        )
        current_cwd = config.get("terminal", {}).get("cwd", "")
        cwd = prompt("  Messaging working directory", current_cwd or str(Path.home()))
        if cwd:
            config["terminal"]["cwd"] = cwd

        # Sudo support
        print()
        existing_sudo = get_env_value("SUDO_PASSWORD")
        if existing_sudo:
            print_info("Sudo 비밀번호: 설정되어 있음")
        else:
            if prompt_yes_no(
                "sudo 지원을 활성화할까요? (apt install 등에 사용할 비밀번호 저장)", False
            ):
                sudo_pass = prompt("  Sudo 비밀번호", password=True)
                if sudo_pass:
                    save_env_value("SUDO_PASSWORD", sudo_pass)
                    print_success("Sudo 비밀번호 저장 완료")

    elif selected_backend == "docker":
        print_success("터미널 백엔드: Docker")

        # Check if Docker is available
        docker_bin = shutil.which("docker")
        if not docker_bin:
            print_warning("PATH에서 Docker를 찾을 수 없습니다!")
            print_info("Docker 설치: https://docs.docker.com/get-docker/")
        else:
            print_info(f"Docker 찾음: {docker_bin}")

        # Docker image
        current_image = config.get("terminal", {}).get(
            "docker_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )
        image = prompt("  Docker image", current_image)
        config["terminal"]["docker_image"] = image
        save_env_value("TERMINAL_DOCKER_IMAGE", image)

        _prompt_container_resources(config)

    elif selected_backend == "singularity":
        print_success("터미널 백엔드: Singularity/Apptainer")

        # Check if singularity/apptainer is available
        sing_bin = shutil.which("apptainer") or shutil.which("singularity")
        if not sing_bin:
            print_warning("PATH에서 Singularity/Apptainer를 찾을 수 없습니다!")
            print_info(
                "설치: https://apptainer.org/docs/admin/main/installation.html"
            )
        else:
            print_info(f"찾음: {sing_bin}")

        current_image = config.get("terminal", {}).get(
            "singularity_image", "docker://nikolaik/python-nodejs:python3.11-nodejs20"
        )
        image = prompt("  컨테이너 이미지", current_image)
        config["terminal"]["singularity_image"] = image
        save_env_value("TERMINAL_SINGULARITY_IMAGE", image)

        _prompt_container_resources(config)

    elif selected_backend == "modal":
        print_success("터미널 백엔드: Modal")
        print_info("서버리스 클라우드 샌드박스입니다. 세션마다 전용 컨테이너가 생성됩니다.")
        from tools.managed_tool_gateway import is_managed_tool_gateway_ready
        from tools.tool_backend_helpers import normalize_modal_mode

        managed_modal_available = bool(
            managed_nous_tools_enabled()
            and
            get_nous_subscription_features(config).nous_auth_present
            and is_managed_tool_gateway_ready("modal")
        )
        modal_mode = normalize_modal_mode(config.get("terminal", {}).get("modal_mode"))
        use_managed_modal = False
        if managed_modal_available:
            modal_choices = [
                "내 Nous 구독 사용",
                "내 Modal 계정 사용",
            ]
            if modal_mode == "managed":
                default_modal_idx = 0
            elif modal_mode == "direct":
                default_modal_idx = 1
            else:
                default_modal_idx = 1 if get_env_value("MODAL_TOKEN_ID") else 0
            modal_mode_idx = prompt_choice(
                "Modal 실행 과금 방식을 선택하세요:",
                modal_choices,
                default_modal_idx,
            )
            use_managed_modal = modal_mode_idx == 0

        if use_managed_modal:
            config["terminal"]["modal_mode"] = "managed"
            print_info("Modal 실행은 관리형 Nous gateway를 사용하며 비용은 구독에 청구됩니다.")
            if get_env_value("MODAL_TOKEN_ID") or get_env_value("MODAL_TOKEN_SECRET"):
                print_info(
                    "직접 Modal 자격 증명도 설정되어 있지만, 이 백엔드는 현재 managed 모드로 고정됩니다."
                )
        else:
            config["terminal"]["modal_mode"] = "direct"
            print_info("Modal 계정이 필요합니다: https://modal.com")

            # Check if modal SDK is installed
            try:
                __import__("modal")
            except ImportError:
                print_info("modal SDK 설치 중...")
                import subprocess

                uv_bin = shutil.which("uv")
                if uv_bin:
                    result = subprocess.run(
                        [
                            uv_bin,
                            "pip",
                            "install",
                            "--python",
                            sys.executable,
                            "modal",
                        ],
                        capture_output=True,
                        text=True,
                    )
                else:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "modal"],
                        capture_output=True,
                        text=True,
                    )
                if result.returncode == 0:
                    print_success("modal SDK 설치 완료")
                else:
                    print_warning("설치 실패 — 직접 실행하세요: pip install modal")

            # Modal token
            print()
            print_info("Modal 인증:")
            print_info("  토큰 발급 위치: https://modal.com/settings")
            existing_token = get_env_value("MODAL_TOKEN_ID")
            if existing_token:
                print_info("  Modal 토큰: 이미 설정되어 있음")
                if prompt_yes_no("  Modal 자격 증명을 업데이트할까요?", False):
                    token_id = prompt("    Modal Token ID", password=True)
                    token_secret = prompt("    Modal Token Secret", password=True)
                    if token_id:
                        save_env_value("MODAL_TOKEN_ID", token_id)
                    if token_secret:
                        save_env_value("MODAL_TOKEN_SECRET", token_secret)
            else:
                token_id = prompt("    Modal Token ID", password=True)
                token_secret = prompt("    Modal Token Secret", password=True)
                if token_id:
                    save_env_value("MODAL_TOKEN_ID", token_id)
                if token_secret:
                    save_env_value("MODAL_TOKEN_SECRET", token_secret)

        _prompt_container_resources(config)

    elif selected_backend == "daytona":
        print_success("터미널 백엔드: Daytona")
        print_info("Daytona 클라우드 개발 환경입니다.")
        print_info("세션마다 파일시스템이 유지되는 전용 샌드박스를 제공합니다.")
        print_info("가입: https://daytona.io")

        # Check if daytona SDK is installed
        try:
            __import__("daytona")
        except ImportError:
            print_info("daytona SDK 설치 중...")
            import subprocess

            uv_bin = shutil.which("uv")
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, "daytona"],
                    capture_output=True,
                    text=True,
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "daytona"],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                print_success("daytona SDK 설치 완료")
            else:
                print_warning("설치 실패 — 직접 실행하세요: pip install daytona")
                if result.stderr:
                    print_info(f"  오류: {result.stderr.strip().splitlines()[-1]}")

        # Daytona API key
        print()
        existing_key = get_env_value("DAYTONA_API_KEY")
        if existing_key:
            print_info("  Daytona API 키: 이미 설정되어 있음")
            if prompt_yes_no("  API 키를 업데이트할까요?", False):
                api_key = prompt("    Daytona API 키", password=True)
                if api_key:
                    save_env_value("DAYTONA_API_KEY", api_key)
                    print_success("    업데이트 완료")
        else:
            api_key = prompt("    Daytona API 키", password=True)
            if api_key:
                save_env_value("DAYTONA_API_KEY", api_key)
                print_success("    설정 완료")

        # Daytona image
        current_image = config.get("terminal", {}).get(
            "daytona_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )
        image = prompt("  샌드박스 이미지", current_image)
        config["terminal"]["daytona_image"] = image
        save_env_value("TERMINAL_DAYTONA_IMAGE", image)

        _prompt_container_resources(config)

    elif selected_backend == "ssh":
        print_success("터미널 백엔드: SSH")
        print_info("원격 머신에서 SSH로 명령어를 실행합니다.")

        # SSH host
        current_host = get_env_value("TERMINAL_SSH_HOST") or ""
        host = prompt("  SSH 호스트 (hostname 또는 IP)", current_host)
        if host:
            save_env_value("TERMINAL_SSH_HOST", host)

        # SSH user
        current_user = get_env_value("TERMINAL_SSH_USER") or ""
        user = prompt("  SSH 사용자", current_user or os.getenv("USER", ""))
        if user:
            save_env_value("TERMINAL_SSH_USER", user)

        # SSH port
        current_port = get_env_value("TERMINAL_SSH_PORT") or "22"
        port = prompt("  SSH 포트", current_port)
        if port and port != "22":
            save_env_value("TERMINAL_SSH_PORT", port)

        # SSH key
        current_key = get_env_value("TERMINAL_SSH_KEY") or ""
        default_key = str(Path.home() / ".ssh" / "id_rsa")
        ssh_key = prompt("  SSH 개인키 경로", current_key or default_key)
        if ssh_key:
            save_env_value("TERMINAL_SSH_KEY", ssh_key)

        # Test connection
        if host and prompt_yes_no("  SSH 연결을 테스트할까요?", True):
            print_info("  연결 테스트 중...")
            import subprocess

            ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
            if ssh_key:
                ssh_cmd.extend(["-i", ssh_key])
            if port and port != "22":
                ssh_cmd.extend(["-p", port])
            ssh_cmd.append(f"{user}@{host}" if user else host)
            ssh_cmd.append("echo ok")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_success("  SSH 연결 성공!")
            else:
                print_warning(f"  SSH 연결 실패: {result.stderr.strip()}")
                print_info("  SSH 키와 호스트 설정을 확인하세요.")

    # Sync terminal backend to .env so terminal_tool picks it up directly.
    # config.yaml is the source of truth, but terminal_tool reads TERMINAL_ENV.
    save_env_value("TERMINAL_ENV", selected_backend)
    if selected_backend == "modal":
        save_env_value("TERMINAL_MODAL_MODE", config["terminal"].get("modal_mode", "auto"))
    save_config(config)
    print()
    print_success(f"터미널 백엔드 설정 완료: {selected_backend}")


# =============================================================================
# Section 3: Agent Settings
# =============================================================================


def _apply_default_agent_settings(config: dict):
    """Apply recommended defaults for all agent settings without prompting."""
    config.setdefault("agent", {})["max_turns"] = 90
    save_env_value("HERMES_MAX_ITERATIONS", "90")

    config.setdefault("display", {})["tool_progress"] = "all"

    config.setdefault("compression", {})["enabled"] = True
    config["compression"]["threshold"] = 0.50

    config.setdefault("session_reset", {}).update({
        "mode": "both",
        "idle_minutes": 1440,
        "at_hour": 4,
    })

    save_config(config)
    print_success("권장 기본값을 적용했습니다:")
    print_info("  최대 반복 수: 90")
    print_info("  도구 진행 표시: all")
    print_info("  압축 임계값: 0.50")
    print_info("  세션 리셋: 비활성 1440분 + 매일 4:00")
    print_info("  나중에 `hermes setup agent`를 실행해 세부 조정할 수 있습니다.")


def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display, compression, session reset."""

    print_header("에이전트 설정")
    print_info(f"   Guide: {_DOCS_BASE}/user-guide/configuration")
    print()

    # ── Max Iterations ──
    current_max = get_env_value("HERMES_MAX_ITERATIONS") or str(
        config.get("agent", {}).get("max_turns", 90)
    )
    print_info("대화 한 번에 도구를 호출할 수 있는 최대 반복 횟수입니다.")
    print_info("값이 높을수록 더 복잡한 작업을 처리할 수 있지만 토큰 비용도 늘어납니다.")
    print_info("기본값 90이면 대부분 충분하고, 열린 탐색 작업은 150+도 고려할 수 있습니다.")

    max_iter_str = prompt("최대 반복 수", current_max)
    try:
        max_iter = int(max_iter_str)
        if max_iter > 0:
            save_env_value("HERMES_MAX_ITERATIONS", str(max_iter))
            config.setdefault("agent", {})["max_turns"] = max_iter
            config.pop("max_turns", None)
            print_success(f"최대 반복 수를 {max_iter}(으)로 설정했습니다")
    except ValueError:
        print_warning("잘못된 숫자입니다. 현재 값을 유지합니다")

    # ── Tool Progress Display ──
    print_info("")
    print_info("도구 진행 표시")
    print_info("도구 활동을 어느 정도까지 보여줄지 결정합니다. (CLI와 메시징 공통)")
    print_info("  off     — 조용히, 최종 응답만 표시")
    print_info("  new     — 도구 이름이 바뀔 때만 표시 (덜 시끄러움)")
    print_info("  all     — 모든 도구 호출을 짧은 미리보기와 함께 표시")
    print_info("  verbose — 전체 인자, 결과, 디버그 로그 표시")

    current_mode = config.get("display", {}).get("tool_progress", "all")
    mode = prompt("도구 진행 표시 모드", current_mode)
    if mode.lower() in ("off", "new", "all", "verbose"):
        if "display" not in config:
            config["display"] = {}
        config["display"]["tool_progress"] = mode.lower()
        save_config(config)
        print_success(f"도구 진행 표시를 다음으로 설정했습니다: {mode.lower()}")
    else:
        print_warning(f"알 수 없는 모드 '{mode}' 입니다. '{current_mode}'을(를) 유지합니다")

    # ── Context Compression ──
    print_header("컨텍스트 압축")
    print_info("컨텍스트가 너무 길어지면 오래된 메시지를 자동으로 요약합니다.")
    print_info(
        "임계값이 높을수록 늦게 압축하고(더 많은 컨텍스트 사용), 낮을수록 빨리 압축합니다."
    )

    config.setdefault("compression", {})["enabled"] = True

    current_threshold = config.get("compression", {}).get("threshold", 0.50)
    threshold_str = prompt("압축 임계값 (0.5-0.95)", str(current_threshold))
    try:
        threshold = float(threshold_str)
        if 0.5 <= threshold <= 0.95:
            config["compression"]["threshold"] = threshold
    except ValueError:
        pass

    print_success(
        f"컨텍스트 압축 임계값을 {config['compression'].get('threshold', 0.50)}(으)로 설정했습니다"
    )

    # ── Session Reset Policy ──
    print_header("세션 리셋 정책")
    print_info(
        "메시징 세션(Telegram, Discord 등)은 시간이 지날수록 컨텍스트가 누적됩니다."
    )
    print_info(
        "메시지를 보낼수록 대화 기록이 길어지고, 그만큼 API 비용도 커질 수 있습니다."
    )
    print_info("")
    print_info(
        "이를 관리하기 위해, 일정 시간 비활성 상태가 지속되거나 매일 정해진 시각에"
    )
    print_info(
        "세션을 자동 리셋하도록 설정할 수 있습니다. 리셋 시 중요한 정보는 먼저"
    )
    print_info(
        "지속 메모리에 저장되지만, 대화 컨텍스트 자체는 비워집니다."
    )
    print_info("")
    print_info("채팅에서 /reset을 입력해 언제든 수동으로 리셋할 수도 있습니다.")
    print_info("")

    reset_choices = [
        "비활성 + 매일 리셋 (권장 - 먼저 도달한 조건으로 리셋)",
        "비활성만 (N분 동안 메시지가 없으면 리셋)",
        "매일만 (매일 고정 시각에 리셋)",
        "자동 리셋 안 함 (/reset 또는 컨텍스트 압축 전까지 유지)",
        "현재 설정 유지",
    ]

    current_policy = config.get("session_reset", {})
    current_mode = current_policy.get("mode", "both")
    current_idle = current_policy.get("idle_minutes", 1440)
    current_hour = current_policy.get("at_hour", 4)

    default_reset = {"both": 0, "idle": 1, "daily": 2, "none": 3}.get(current_mode, 0)

    reset_idx = prompt_choice("세션 리셋 모드:", reset_choices, default_reset)

    config.setdefault("session_reset", {})

    if reset_idx == 0:  # Both
        config["session_reset"]["mode"] = "both"
        idle_str = prompt("  비활성 타임아웃 (분)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config["session_reset"]["idle_minutes"] = idle_val
        except ValueError:
            pass
        hour_str = prompt("  일일 리셋 시각 (0-23, 로컬 시간)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config["session_reset"]["at_hour"] = hour_val
        except ValueError:
            pass
        print_success(
            f"세션은 {config['session_reset'].get('idle_minutes', 1440)}분 비활성 후 또는 매일 {config['session_reset'].get('at_hour', 4)}:00에 리셋됩니다"
        )
    elif reset_idx == 1:  # Idle only
        config["session_reset"]["mode"] = "idle"
        idle_str = prompt("  비활성 타임아웃 (분)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config["session_reset"]["idle_minutes"] = idle_val
        except ValueError:
            pass
        print_success(
            f"세션은 {config['session_reset'].get('idle_minutes', 1440)}분 비활성 상태가 되면 리셋됩니다"
        )
    elif reset_idx == 2:  # Daily only
        config["session_reset"]["mode"] = "daily"
        hour_str = prompt("  일일 리셋 시각 (0-23, 로컬 시간)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config["session_reset"]["at_hour"] = hour_val
        except ValueError:
            pass
        print_success(
            f"세션은 매일 {config['session_reset'].get('at_hour', 4)}:00에 리셋됩니다"
        )
    elif reset_idx == 3:  # None
        config["session_reset"]["mode"] = "none"
        print_info(
            "세션은 자동으로 리셋되지 않습니다. 컨텍스트 관리는 압축만으로 처리됩니다."
        )
        print_warning(
            "긴 대화는 비용이 계속 늘어날 수 있습니다. 필요할 때 /reset을 수동으로 사용하세요."
        )
    # else: keep current (idx == 4)

    save_config(config)


# =============================================================================
# Section 4: Messaging Platforms (Gateway)
# =============================================================================


def _setup_telegram():
    """Configure Telegram bot credentials and allowlist."""
    print_header("Telegram")
    existing = get_env_value("TELEGRAM_BOT_TOKEN")
    if existing:
        print_info("Telegram: 이미 설정되어 있음")
        if not prompt_yes_no("Telegram을 다시 설정할까요?", False):
            # Check missing allowlist on existing config
            if not get_env_value("TELEGRAM_ALLOWED_USERS"):
                print_info("⚠️  Telegram에 사용자 허용 목록이 없습니다 - 누구나 봇을 사용할 수 있어요!")
                if prompt_yes_no("허용할 사용자를 지금 추가할까요?", True):
                    print_info("   Telegram 사용자 ID 확인: @userinfobot에 메시지 보내기")
                    allowed_users = prompt("허용할 사용자 ID (쉼표로 구분)")
                    if allowed_users:
                        save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
                        print_success("Telegram allowlist 설정 완료")
            return

    print_info("@BotFather로 Telegram 봇 만들기")
    token = prompt("Telegram 봇 토큰", password=True)
    if not token:
        return
    save_env_value("TELEGRAM_BOT_TOKEN", token)
    print_success("Telegram 토큰 저장 완료")

    print()
    print_info("🔒 보안: 누가 봇을 사용할 수 있는지 제한하세요")
    print_info("   Telegram 사용자 ID 확인 방법:")
    print_info("   1. Telegram에서 @userinfobot에 메시지 보내기")
    print_info("   2. 숫자 ID(예: 123456789)를 답장으로 받습니다")
    print()
    allowed_users = prompt(
        "허용할 사용자 ID (쉼표로 구분, 비워두면 누구나 접근 가능)"
    )
    if allowed_users:
        save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Telegram allowlist 설정 완료 - 목록에 있는 사용자만 봇을 사용할 수 있습니다")
    else:
        print_info("⚠️  allowlist가 없습니다 - 봇을 찾는 누구나 사용할 수 있습니다!")

    print()
    print_info("📬 홈 채널: Hermes가 cron 결과, 플랫폼 간 메시지, 알림을 전달하는 곳입니다.")
    print_info("   Telegram DM에서는 위와 같은 사용자 ID를 사용합니다.")

    first_user_id = allowed_users.split(",")[0].strip() if allowed_users else ""
    if first_user_id:
        if prompt_yes_no(f"Use your user ID ({first_user_id}) as the home channel?", True):
            save_env_value("TELEGRAM_HOME_CHANNEL", first_user_id)
            print_success(f"Telegram 홈 채널을 {first_user_id}(으)로 설정했습니다")
        else:
            home_channel = prompt("홈 채널 ID (또는 비워두고 나중에 Telegram에서 /set-home으로 설정)")
            if home_channel:
                save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)
    else:
        print_info("   나중에 Telegram 채팅에서 /set-home을 입력해 설정할 수도 있습니다.")
        home_channel = prompt("홈 채널 ID (비워두면 나중에 설정)")
        if home_channel:
            save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)


def _setup_discord():
    """Configure Discord bot credentials and allowlist."""
    print_header("Discord")
    existing = get_env_value("DISCORD_BOT_TOKEN")
    if existing:
        print_info("Discord: 이미 설정되어 있음")
        if not prompt_yes_no("Discord를 다시 설정할까요?", False):
            if not get_env_value("DISCORD_ALLOWED_USERS"):
                print_info("⚠️  Discord에 사용자 허용 목록이 없습니다 - 누구나 봇을 사용할 수 있어요!")
                if prompt_yes_no("허용할 사용자를 지금 추가할까요?", True):
                    print_info("   Discord ID 확인: 개발자 모드를 켜고 이름을 우클릭 → ID 복사")
                    allowed_users = prompt("허용할 사용자 ID (쉼표로 구분)")
                    if allowed_users:
                        cleaned_ids = _clean_discord_user_ids(allowed_users)
                        save_env_value("DISCORD_ALLOWED_USERS", ",".join(cleaned_ids))
                        print_success("Discord allowlist 설정 완료")
            return

    print_info("Discord 개발자 포털에서 봇 만들기: https://discord.com/developers/applications")
    token = prompt("Discord 봇 토큰", password=True)
    if not token:
        return
    save_env_value("DISCORD_BOT_TOKEN", token)
    print_success("Discord 토큰 저장 완료")

    print()
    print_info("🔒 보안: 누가 봇을 사용할 수 있는지 제한하세요")
    print_info("   Discord 사용자 ID 확인 방법:")
    print_info("   1. Discord 설정에서 개발자 모드 활성화")
    print_info("   2. 내 이름을 우클릭 → ID 복사")
    print()
    print_info("   gateway 시작 시 Discord 사용자명도 사용할 수 있습니다.")
    print()
    allowed_users = prompt(
        "허용할 사용자 ID 또는 사용자명 (쉼표로 구분, 비워두면 누구나 접근 가능)"
    )
    if allowed_users:
        cleaned_ids = _clean_discord_user_ids(allowed_users)
        save_env_value("DISCORD_ALLOWED_USERS", ",".join(cleaned_ids))
        print_success("Discord allowlist 설정 완료")
    else:
        print_info("⚠️  allowlist가 없습니다 - 봇이 있는 서버의 누구나 사용할 수 있습니다!")

    print()
    print_info("📬 홈 채널: Hermes가 cron 결과, 플랫폼 간 메시지, 알림을 전달하는 곳입니다.")
    print_info("   채널 ID 확인: 채널 우클릭 → 채널 ID 복사")
    print_info("   (Discord 설정에서 개발자 모드가 필요합니다)")
    print_info("   나중에 Discord 채널에서 /set-home을 입력해 설정할 수도 있습니다.")
    home_channel = prompt("홈 채널 ID (비워두면 나중에 /set-home으로 설정)")
    if home_channel:
        save_env_value("DISCORD_HOME_CHANNEL", home_channel)


def _clean_discord_user_ids(raw: str) -> list:
    """Strip common Discord mention prefixes from a comma-separated ID string."""
    cleaned = []
    for uid in raw.replace(" ", "").split(","):
        uid = uid.strip()
        if uid.startswith("<@") and uid.endswith(">"):
            uid = uid.lstrip("<@!").rstrip(">")
        if uid.lower().startswith("user:"):
            uid = uid[5:]
        if uid:
            cleaned.append(uid)
    return cleaned


def _setup_slack():
    """Configure Slack bot credentials."""
    print_header("Slack")
    existing = get_env_value("SLACK_BOT_TOKEN")
    if existing:
        print_info("Slack: 이미 설정되어 있음")
        if not prompt_yes_no("Slack을 다시 설정할까요?", False):
            return

    print_info("Slack 앱 생성 단계:")
    print_info("   1. https://api.slack.com/apps 에서 New App 생성 (from scratch)")
    print_info("   2. Socket Mode 활성화: Settings → Socket Mode → Enable")
    print_info("      • 'connections:write' scope로 App-Level Token 생성")
    print_info("   3. Bot Token Scopes 추가: Features → OAuth & Permissions")
    print_info("      필수 scope: chat:write, app_mentions:read,")
    print_info("      channels:history, channels:read, im:history,")
    print_info("      im:read, im:write, users:read, files:read, files:write")
    print_info("      비공개 채널용 선택 scope: groups:history")
    print_info("   4. 이벤트 구독: Features → Event Subscriptions → Enable")
    print_info("      필수 이벤트: message.im, message.channels, app_mention")
    print_info("      비공개 채널용 선택 이벤트: message.groups")
    print_warning("   ⚠ message.channels scope가 없으면 봇은 DM에서만 동작하고,")
    print_warning("     공개 채널에서는 동작하지 않습니다.")
    print_info("   5. Workspace에 설치: Settings → Install App")
    print_info("   6. scope나 event를 바꾼 뒤에는 앱을 다시 설치하세요")
    print_info("   7. After installing, invite the bot to channels: /invite @YourBot")
    print()
    print_info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/slack/")
    print()
    bot_token = prompt("Slack Bot Token (xoxb-...)", password=True)
    if not bot_token:
        return
    save_env_value("SLACK_BOT_TOKEN", bot_token)
    app_token = prompt("Slack App Token (xapp-...)", password=True)
    if app_token:
        save_env_value("SLACK_APP_TOKEN", app_token)
    print_success("Slack tokens saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find a Member ID: click a user's name → View full profile → ⋮ → Copy member ID")
    print()
    allowed_users = prompt(
        "허용할 사용자 ID (쉼표로 구분, 비워두면 페어링된 사용자 외에는 모두 거부)"
    )
    if allowed_users:
        save_env_value("SLACK_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Slack allowlist 설정 완료")
    else:
        print_warning("⚠️  Slack allowlist가 없습니다 - 페어링되지 않은 사용자는 기본적으로 거부됩니다.")
        print_info("   진짜로 워크스페이스 전체 접근을 열려면")
        print_info("   SLACK_ALLOW_ALL_USERS=true 또는 GATEWAY_ALLOW_ALL_USERS=true 를 설정하세요.")


def _setup_matrix():
    """Configure Matrix credentials."""
    print_header("Matrix")
    existing = get_env_value("MATRIX_ACCESS_TOKEN") or get_env_value("MATRIX_PASSWORD")
    if existing:
        print_info("Matrix: 이미 설정되어 있음")
        if not prompt_yes_no("Matrix를 다시 설정할까요?", False):
            return

    print_info("어떤 Matrix homeserver와도 사용할 수 있습니다. (Synapse, Conduit, Dendrite, matrix.org)")
    print_info("   1. homeserver에 봇 사용자를 만들거나 본인 계정을 사용하세요")
    print_info("   2. Element에서 access token을 얻거나 사용자 ID + 비밀번호를 입력하세요")
    print()
    homeserver = prompt("Homeserver URL (e.g. https://matrix.example.org)")
    if homeserver:
        save_env_value("MATRIX_HOMESERVER", homeserver.rstrip("/"))

    print()
    print_info("Auth: provide an access token (recommended), or user ID + password.")
    token = prompt("Access token (leave empty for password login)", password=True)
    if token:
        save_env_value("MATRIX_ACCESS_TOKEN", token)
        user_id = prompt("User ID (@bot:server — optional, will be auto-detected)")
        if user_id:
            save_env_value("MATRIX_USER_ID", user_id)
        print_success("Matrix access token saved")
    else:
        user_id = prompt("User ID (@bot:server)")
        if user_id:
            save_env_value("MATRIX_USER_ID", user_id)
        password = prompt("Password", password=True)
        if password:
            save_env_value("MATRIX_PASSWORD", password)
            print_success("Matrix credentials saved")

    if token or get_env_value("MATRIX_PASSWORD"):
        print()
        want_e2ee = prompt_yes_no("종단간 암호화(E2EE)를 활성화할까요?", False)
        if want_e2ee:
            save_env_value("MATRIX_ENCRYPTION", "true")
            print_success("E2EE enabled")

        matrix_pkg = "mautrix[encryption]" if want_e2ee else "mautrix"
        try:
            __import__("mautrix")
        except ImportError:
            print_info(f"Installing {matrix_pkg}...")
            import subprocess
            uv_bin = shutil.which("uv")
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, matrix_pkg],
                    capture_output=True, text=True,
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", matrix_pkg],
                    capture_output=True, text=True,
                )
            if result.returncode == 0:
                print_success(f"{matrix_pkg} installed")
            else:
                print_warning(f"설치 실패 — 직접 실행하세요: pip install '{matrix_pkg}'")
                if result.stderr:
                    print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")

        print()
        print_info("🔒 보안: 누가 봇을 사용할 수 있는지 제한하세요")
        print_info("   Matrix 사용자 ID 형식: @username:server")
        print()
        allowed_users = prompt("허용할 사용자 ID (쉼표로 구분, 비워두면 누구나 접근 가능)")
        if allowed_users:
            save_env_value("MATRIX_ALLOWED_USERS", allowed_users.replace(" ", ""))
            print_success("Matrix allowlist configured")
        else:
            print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")

        print()
        print_info("📬 Home Room: where Hermes delivers cron job results and notifications.")
        print_info("   Room IDs look like !abc123:server (shown in Element room settings)")
        print_info("   You can also set this later by typing /set-home in a Matrix room.")
        home_room = prompt("Home room ID (leave empty to set later with /set-home)")
        if home_room:
            save_env_value("MATRIX_HOME_ROOM", home_room)


def _setup_mattermost():
    """Configure Mattermost bot credentials."""
    print_header("Mattermost")
    existing = get_env_value("MATTERMOST_TOKEN")
    if existing:
        print_info("Mattermost: 이미 설정되어 있음")
        if not prompt_yes_no("Mattermost를 다시 설정할까요?", False):
            return

    print_info("어떤 self-hosted Mattermost 인스턴스와도 사용할 수 있습니다.")
    print_info("   1. Mattermost에서: Integrations → Bot Accounts → Add Bot Account")
    print_info("   2. 봇 토큰을 복사하세요")
    print()
    mm_url = prompt("Mattermost 서버 URL (예: https://mm.example.com)")
    if mm_url:
        save_env_value("MATTERMOST_URL", mm_url.rstrip("/"))
    token = prompt("Bot token", password=True)
    if not token:
        return
    save_env_value("MATTERMOST_TOKEN", token)
    print_success("Mattermost token saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find your user ID: click your avatar → Profile")
    print_info("   or use the API: GET /api/v4/users/me")
    print()
    allowed_users = prompt("허용할 사용자 ID (쉼표로 구분, 비워두면 누구나 접근 가능)")
    if allowed_users:
        save_env_value("MATTERMOST_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Mattermost allowlist configured")
    else:
        print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")

    print()
    print_info("📬 홈 채널: Hermes가 cron 결과와 알림을 전달하는 곳입니다.")
    print_info("   채널 ID 확인: 채널 이름 클릭 → 정보 보기 → ID 복사")
    print_info("   나중에 Mattermost 채널에서 /set-home을 입력해 설정할 수도 있습니다.")
    home_channel = prompt("홈 채널 ID (비워두면 나중에 /set-home으로 설정)")
    if home_channel:
        save_env_value("MATTERMOST_HOME_CHANNEL", home_channel)


def _setup_whatsapp():
    """Configure WhatsApp bridge."""
    print_header("WhatsApp")
    existing = get_env_value("WHATSAPP_ENABLED")
    if existing:
        print_info("WhatsApp: already enabled")
        return

    print_info("WhatsApp은 내장 브리지(Baileys)로 연결됩니다.")
    print_info("Node.js가 필요합니다. 안내형 설정은 'hermes whatsapp'을 실행하세요.")
    print()
    if prompt_yes_no("지금 WhatsApp을 활성화할까요?", True):
        save_env_value("WHATSAPP_ENABLED", "true")
        print_success("WhatsApp 활성화 완료")
        print_info("'hermes whatsapp'을 실행해 모드(별도 봇 번호 또는")
        print_info("개인 셀프 채팅)를 선택하고 QR 코드로 페어링하세요.")


def _setup_weixin():
    """Configure Weixin (personal WeChat) via iLink Bot API QR login."""
    from hermes_cli.gateway import _setup_weixin as _gateway_setup_weixin
    _gateway_setup_weixin()


def _setup_signal():
    """Configure Signal via gateway setup."""
    from hermes_cli.gateway import _setup_signal as _gateway_setup_signal
    _gateway_setup_signal()


def _setup_email():
    """Configure Email via gateway setup."""
    from hermes_cli.gateway import _setup_email as _gateway_setup_email
    _gateway_setup_email()


def _setup_sms():
    """Configure SMS (Twilio) via gateway setup."""
    from hermes_cli.gateway import _setup_sms as _gateway_setup_sms
    _gateway_setup_sms()


def _setup_dingtalk():
    """Configure DingTalk via gateway setup."""
    from hermes_cli.gateway import _setup_dingtalk as _gateway_setup_dingtalk
    _gateway_setup_dingtalk()


def _setup_feishu():
    """Configure Feishu / Lark via gateway setup."""
    from hermes_cli.gateway import _setup_feishu as _gateway_setup_feishu
    _gateway_setup_feishu()


def _setup_wecom():
    """Configure WeCom (Enterprise WeChat) via gateway setup."""
    from hermes_cli.gateway import _setup_wecom as _gateway_setup_wecom
    _gateway_setup_wecom()


def _setup_wecom_callback():
    """Configure WeCom Callback (self-built app) via gateway setup."""
    from hermes_cli.gateway import _setup_wecom_callback as _gw_setup
    _gw_setup()


def _setup_qqbot():
    """Configure QQ Bot gateway."""
    print_header("QQ Bot")
    existing = get_env_value("QQ_APP_ID")
    if existing:
        print_info("QQ Bot: 이미 설정되어 있음")
        if not prompt_yes_no("QQ Bot을 다시 설정할까요?", False):
            return

    print_info("공식 QQ Bot API (v2)를 통해 Hermes를 QQ에 연결합니다.")
    print_info("   q.qq.com 에서 QQ Bot 애플리케이션이 필요합니다")
    print_info("   참고: https://bot.q.qq.com/wiki/develop/api-v2/")
    print()

    app_id = prompt("QQ Bot App ID")
    if not app_id:
        print_warning("App ID가 필요합니다 — QQ Bot 설정을 건너뜁니다")
        return
    save_env_value("QQ_APP_ID", app_id.strip())

    client_secret = prompt("QQ Bot App Secret", password=True)
    if not client_secret:
        print_warning("App Secret이 필요합니다 — QQ Bot 설정을 건너뜁니다")
        return
    save_env_value("QQ_CLIENT_SECRET", client_secret)
    print_success("QQ Bot 자격 증명 저장 완료")

    print()
    print_info("🔒 보안: 누가 봇에 DM을 보낼 수 있는지 제한하세요")
    print_info("   QQ 사용자 OpenID를 사용하세요 (이벤트 payload에서 확인 가능)")
    print()
    allowed_users = prompt("허용할 사용자 OpenID (쉼표로 구분, 비워두면 누구나 접근 가능)")
    if allowed_users:
        save_env_value("QQ_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("QQ Bot allowlist 설정 완료")
    else:
        print_info("⚠️  allowlist가 없습니다 — 누구나 봇에 DM을 보낼 수 있습니다!")

    print()
    print_info("📬 홈 채널: cron 전달과 알림에 사용할 OpenID입니다.")
    home_channel = prompt("홈 채널 OpenID (비워두면 나중에 설정)")
    if home_channel:
        save_env_value("QQ_HOME_CHANNEL", home_channel)

    print()
    print_success("QQ Bot configured!")


def _setup_bluebubbles():
    """Configure BlueBubbles iMessage gateway."""
    print_header("BlueBubbles (iMessage)")
    existing = get_env_value("BLUEBUBBLES_SERVER_URL")
    if existing:
        print_info("BlueBubbles: 이미 설정되어 있음")
        if not prompt_yes_no("BlueBubbles를 다시 설정할까요?", False):
            return

    print_info("BlueBubbles를 통해 Hermes를 iMessage에 연결합니다 — 무료 오픈소스")
    print_info("macOS 서버가 iMessage를 다른 기기에 브리지합니다.")
    print_info("   BlueBubbles Server v1.0.0+가 실행 중인 Mac이 필요합니다")
    print_info("   다운로드: https://bluebubbles.app/")
    print()
    print_info("BlueBubbles Server → Settings → API 에서 Server URL과 Password를 확인하세요.")
    print()

    server_url = prompt("BlueBubbles 서버 URL (예: http://192.168.1.10:1234)")
    if not server_url:
        print_warning("서버 URL이 필요합니다 — BlueBubbles 설정을 건너뜁니다")
        return
    save_env_value("BLUEBUBBLES_SERVER_URL", server_url.rstrip("/"))

    password = prompt("BlueBubbles 서버 비밀번호", password=True)
    if not password:
        print_warning("비밀번호가 필요합니다 — BlueBubbles 설정을 건너뜁니다")
        return
    save_env_value("BLUEBUBBLES_PASSWORD", password)
    print_success("BlueBubbles 자격 증명 저장 완료")

    print()
    print_info("🔒 보안: 누가 봇에 메시지를 보낼 수 있는지 제한하세요")
    print_info("   iMessage 주소를 사용하세요: 이메일(user@icloud.com) 또는 전화번호(+155****4567)")
    print()
    allowed_users = prompt("허용할 iMessage 주소 (쉼표로 구분, 비워두면 누구나 접근 가능)")
    if allowed_users:
        save_env_value("BLUEBUBBLES_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("BlueBubbles allowlist configured")
    else:
        print_info("⚠️  No allowlist set — anyone who can iMessage you can use the bot!")

    print()
    print_info("📬 홈 채널: cron 전달과 알림에 사용할 전화번호 또는 이메일입니다.")
    print_info("   나중에 iMessage 채팅에서 /set-home으로도 설정할 수 있습니다.")
    home_channel = prompt("홈 채널 주소 (비워두면 나중에 설정)")
    if home_channel:
        save_env_value("BLUEBUBBLES_HOME_CHANNEL", home_channel)

    print()
    print_info("고급 설정 (대부분 기본값이면 충분합니다):")
    if prompt_yes_no("Webhook listener settings를 설정할까요?", False):
        webhook_port = prompt("Webhook listener 포트 (기본값: 8645)")
        if webhook_port:
            try:
                save_env_value("BLUEBUBBLES_WEBHOOK_PORT", str(int(webhook_port)))
                print_success(f"Webhook 포트를 {webhook_port}로 설정했습니다")
            except ValueError:
                print_warning("잘못된 포트 번호입니다. 기본값 8645를 사용합니다")

    print()
    print_info("BlueBubbles Private API helper가 있으면 타이핑 표시,")
    print_info("읽음 확인, tapback 반응을 지원합니다. 기본 메시징은 없어도 동작합니다.")
    print_info("   Install: https://docs.bluebubbles.app/helper-bundle/installation")


def _setup_qqbot():
    """Configure QQ Bot (Official API v2) via standard platform setup."""
    from hermes_cli.gateway import _PLATFORMS
    qq_platform = next((p for p in _PLATFORMS if p["key"] == "qqbot"), None)
    if qq_platform:
        from hermes_cli.gateway import _setup_standard_platform
        _setup_standard_platform(qq_platform)


def _setup_webhooks():
    """Configure webhook integration."""
    print_header("Webhooks")
    existing = get_env_value("WEBHOOK_ENABLED")
    if existing:
        print_info("Webhooks: 이미 설정되어 있음")
        if not prompt_yes_no("Webhooks를 다시 설정할까요?", False):
            return

    print()
    print_warning("⚠  Webhook and SMS platforms require exposing gateway ports to the")
    print_warning("   internet. For security, run the gateway in a sandboxed environment")
    print_warning("   (Docker, VM, etc.) to limit blast radius from prompt injection.")
    print()
    print_info("   Full guide: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/")
    print()

    port = prompt("Webhook port (default 8644)")
    if port:
        try:
            save_env_value("WEBHOOK_PORT", str(int(port)))
            print_success(f"Webhook 포트를 {port}로 설정했습니다")
        except ValueError:
            print_warning("잘못된 포트 번호입니다. 기본값 8644를 사용합니다")

    secret = prompt("전역 HMAC 시크릿 (모든 라우트에서 공유)", password=True)
    if secret:
        save_env_value("WEBHOOK_SECRET", secret)
        print_success("Webhook secret saved")
    else:
        print_warning("No secret set — you must configure per-route secrets in config.yaml")

    save_env_value("WEBHOOK_ENABLED", "true")
    print()
    print_success("Webhooks 활성화 완료! 다음 단계:")
    from hermes_constants import display_hermes_home as _dhh
    print_info(f"   1. {_dhh()}/config.yaml에서 webhook 라우트를 정의하세요")
    print_info("   2. 서비스(GitHub, GitLab 등)의 대상 주소를 다음으로 설정하세요:")
    print_info("      http://your-server:8644/webhooks/<route-name>")
    print()
    print_info("   라우트 설정 가이드:")
    print_info("   https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/#configuring-routes")
    print()
    print_info("   편집기에서 config 열기:  hermes config edit")


# Platform registry for the gateway checklist
_GATEWAY_PLATFORMS = [
    ("Telegram", "TELEGRAM_BOT_TOKEN", _setup_telegram),
    ("Discord", "DISCORD_BOT_TOKEN", _setup_discord),
    ("Slack", "SLACK_BOT_TOKEN", _setup_slack),
    ("Signal", "SIGNAL_HTTP_URL", _setup_signal),
    ("Email", "EMAIL_ADDRESS", _setup_email),
    ("SMS (Twilio)", "TWILIO_ACCOUNT_SID", _setup_sms),
    ("Matrix", "MATRIX_ACCESS_TOKEN", _setup_matrix),
    ("Mattermost", "MATTERMOST_TOKEN", _setup_mattermost),
    ("WhatsApp", "WHATSAPP_ENABLED", _setup_whatsapp),
    ("DingTalk", "DINGTALK_CLIENT_ID", _setup_dingtalk),
    ("Feishu / Lark", "FEISHU_APP_ID", _setup_feishu),
    ("WeCom (Enterprise WeChat)", "WECOM_BOT_ID", _setup_wecom),
    ("WeCom Callback (Self-Built App)", "WECOM_CALLBACK_CORP_ID", _setup_wecom_callback),
    ("Weixin (WeChat)", "WEIXIN_ACCOUNT_ID", _setup_weixin),
    ("BlueBubbles (iMessage)", "BLUEBUBBLES_SERVER_URL", _setup_bluebubbles),
    ("QQ Bot", "QQ_APP_ID", _setup_qqbot),
    ("Webhooks (GitHub, GitLab, etc.)", "WEBHOOK_ENABLED", _setup_webhooks),
]


def setup_gateway(config: dict):
    """Configure messaging platform integrations."""
    print_header("메시징 플랫폼")
    print_info("어디서든 Hermes와 대화할 수 있도록 메시징 플랫폼에 연결하세요.")
    print_info("Space로 토글하고 Enter로 확인하세요.")
    print()

    # Build checklist items, pre-selecting already-configured platforms
    items = []
    pre_selected = []
    for i, (name, env_var, _func) in enumerate(_GATEWAY_PLATFORMS):
        # Matrix has two possible env vars
        is_configured = bool(get_env_value(env_var))
        if name == "Matrix" and not is_configured:
            is_configured = bool(get_env_value("MATRIX_PASSWORD"))
        label = f"{name}  (configured)" if is_configured else name
        items.append(label)
        if is_configured:
            pre_selected.append(i)

    selected = prompt_checklist("설정할 플랫폼을 선택하세요:", items, pre_selected)

    if not selected:
        print_info("선택한 플랫폼이 없습니다. 나중에 'hermes setup gateway'로 설정하세요.")
        return

    for idx in selected:
        name, _env_var, setup_func = _GATEWAY_PLATFORMS[idx]
        setup_func()

    # ── Gateway Service Setup ──
    any_messaging = (
        get_env_value("TELEGRAM_BOT_TOKEN")
        or get_env_value("DISCORD_BOT_TOKEN")
        or get_env_value("SLACK_BOT_TOKEN")
        or get_env_value("SIGNAL_HTTP_URL")
        or get_env_value("EMAIL_ADDRESS")
        or get_env_value("TWILIO_ACCOUNT_SID")
        or get_env_value("MATTERMOST_TOKEN")
        or get_env_value("MATRIX_ACCESS_TOKEN")
        or get_env_value("MATRIX_PASSWORD")
        or get_env_value("WHATSAPP_ENABLED")
        or get_env_value("DINGTALK_CLIENT_ID")
        or get_env_value("FEISHU_APP_ID")
        or get_env_value("WECOM_BOT_ID")
        or get_env_value("WEIXIN_ACCOUNT_ID")
        or get_env_value("BLUEBUBBLES_SERVER_URL")
        or get_env_value("QQ_APP_ID")
        or get_env_value("WEBHOOK_ENABLED")
    )
    if any_messaging:
        print()
        print_info("━" * 50)
        print_success("Messaging platforms configured!")

        # Check if any home channels are missing
        missing_home = []
        if get_env_value("TELEGRAM_BOT_TOKEN") and not get_env_value(
            "TELEGRAM_HOME_CHANNEL"
        ):
            missing_home.append("Telegram")
        if get_env_value("DISCORD_BOT_TOKEN") and not get_env_value(
            "DISCORD_HOME_CHANNEL"
        ):
            missing_home.append("Discord")
        if get_env_value("SLACK_BOT_TOKEN") and not get_env_value("SLACK_HOME_CHANNEL"):
            missing_home.append("Slack")
        if get_env_value("BLUEBUBBLES_SERVER_URL") and not get_env_value("BLUEBUBBLES_HOME_CHANNEL"):
            missing_home.append("BlueBubbles")
        if get_env_value("QQ_APP_ID") and not get_env_value("QQ_HOME_CHANNEL"):
            missing_home.append("QQBot")

        if missing_home:
            print()
            print_warning(f"No home channel set for: {', '.join(missing_home)}")
            print_info("   Without a home channel, cron jobs and cross-platform")
            print_info("   messages can't be delivered to those platforms.")
            print_info("   Set one later with /set-home in your chat, or:")
            for plat in missing_home:
                print_info(
                    f"     hermes config set {plat.upper()}_HOME_CHANNEL <channel_id>"
                )

        # Offer to install the gateway as a system service
        import platform as _platform

        _is_linux = _platform.system() == "Linux"
        _is_macos = _platform.system() == "Darwin"

        from hermes_cli.gateway import (
            _is_service_installed,
            _is_service_running,
            supports_systemd_services,
            has_conflicting_systemd_units,
            install_linux_gateway_from_setup,
            print_systemd_scope_conflict_warning,
            systemd_start,
            systemd_restart,
            launchd_install,
            launchd_start,
            launchd_restart,
        )

        service_installed = _is_service_installed()
        service_running = _is_service_running()
        supports_systemd = supports_systemd_services()
        supports_service_manager = supports_systemd or _is_macos

        print()
        if supports_systemd and has_conflicting_systemd_units():
            print_systemd_scope_conflict_warning()
            print()

        if service_running:
            if prompt_yes_no("  Restart the gateway to pick up changes?", True):
                try:
                    if supports_systemd:
                        systemd_restart()
                    elif _is_macos:
                        launchd_restart()
                except Exception as e:
                    print_error(f"  Restart failed: {e}")
        elif service_installed:
            if prompt_yes_no("  Start the gateway service?", True):
                try:
                    if supports_systemd:
                        systemd_start()
                    elif _is_macos:
                        launchd_start()
                except Exception as e:
                    print_error(f"  Start failed: {e}")
        elif supports_service_manager:
            svc_name = "systemd" if supports_systemd else "launchd"
            if prompt_yes_no(
                f"  게이트웨이를 {svc_name} 서비스로 설치할까요? (백그라운드 실행, 부팅 시 자동 시작)",
                True,
            ):
                try:
                    installed_scope = None
                    did_install = False
                    if supports_systemd:
                        installed_scope, did_install = install_linux_gateway_from_setup(force=False)
                    else:
                        launchd_install(force=False)
                        did_install = True
                    print()
                    if did_install and prompt_yes_no("  지금 서비스를 시작할까요?", True):
                        try:
                            if supports_systemd:
                                systemd_start(system=installed_scope == "system")
                            elif _is_macos:
                                launchd_start()
                        except Exception as e:
                            print_error(f"  시작 실패: {e}")
                except Exception as e:
                    print_error(f"  설치 실패: {e}")
                    print_info("  수동으로 시도할 수 있습니다: hermes gateway install")
            else:
                print_info("  나중에 설치할 수 있습니다: hermes gateway install")
                if supports_systemd:
                    print_info("  또는 부팅 서비스로: sudo hermes gateway install --system")
                print_info("  또는 포그라운드 실행:  hermes gateway")
        else:
            from hermes_constants import is_container
            if is_container():
                print_info("봇을 온라인으로 올리려면 gateway를 시작하세요:")
                print_info("   hermes gateway run          # 컨테이너의 메인 프로세스로 실행")
                print_info("")
                print_info("자동 재시작에는 Docker restart 정책을 사용하세요:")
                print_info("   docker run --restart unless-stopped ...")
                print_info("   docker restart <container>  # 수동 재시작")
            else:
                print_info("봇을 온라인으로 올리려면 gateway를 시작하세요:")
                print_info("   hermes gateway              # 포그라운드 실행")

        print_info("━" * 50)


# =============================================================================
# Section 5: Tool Configuration (delegates to unified tools_config.py)
# =============================================================================


def setup_tools(config: dict, first_install: bool = False):
    """Configure tools — delegates to the unified tools_command() in tools_config.py.

    Both `hermes setup tools` and `hermes tools` use the same flow:
    platform selection → toolset toggles → provider/API key configuration.

    Args:
        first_install: When True, uses the simplified first-install flow
            (no platform menu, prompts for all unconfigured API keys).
    """
    from hermes_cli.tools_config import tools_command

    tools_command(first_install=first_install, config=config)


# =============================================================================
# Post-Migration Section Skip Logic
# =============================================================================


def _get_section_config_summary(config: dict, section_key: str) -> Optional[str]:
    """Return a short summary if a setup section is already configured, else None.

    Used after OpenClaw migration to detect which sections can be skipped.
    ``get_env_value`` is the module-level import from hermes_cli.config
    so that test patches on ``setup_mod.get_env_value`` take effect.
    """
    if section_key == "model":
        has_key = bool(
            get_env_value("OPENROUTER_API_KEY")
            or get_env_value("OPENAI_API_KEY")
            or get_env_value("ANTHROPIC_API_KEY")
        )
        if not has_key:
            # Check for OAuth providers
            try:
                from hermes_cli.auth import get_active_provider
                if get_active_provider():
                    has_key = True
            except Exception:
                pass
        if not has_key:
            return None
        model = config.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        if isinstance(model, dict):
            return str(model.get("default") or model.get("model") or "configured")
        return "configured"

    elif section_key == "terminal":
        backend = config.get("terminal", {}).get("backend", "local")
        return f"backend: {backend}"

    elif section_key == "agent":
        max_turns = config.get("agent", {}).get("max_turns", 90)
        return f"max turns: {max_turns}"

    elif section_key == "gateway":
        platforms = []
        if get_env_value("TELEGRAM_BOT_TOKEN"):
            platforms.append("Telegram")
        if get_env_value("DISCORD_BOT_TOKEN"):
            platforms.append("Discord")
        if get_env_value("SLACK_BOT_TOKEN"):
            platforms.append("Slack")
        if get_env_value("SIGNAL_ACCOUNT"):
            platforms.append("Signal")
        if get_env_value("EMAIL_ADDRESS"):
            platforms.append("Email")
        if get_env_value("TWILIO_ACCOUNT_SID"):
            platforms.append("SMS")
        if get_env_value("MATRIX_ACCESS_TOKEN") or get_env_value("MATRIX_PASSWORD"):
            platforms.append("Matrix")
        if get_env_value("MATTERMOST_TOKEN"):
            platforms.append("Mattermost")
        if get_env_value("WHATSAPP_PHONE_NUMBER_ID"):
            platforms.append("WhatsApp")
        if get_env_value("DINGTALK_CLIENT_ID"):
            platforms.append("DingTalk")
        if get_env_value("FEISHU_APP_ID"):
            platforms.append("Feishu")
        if get_env_value("WECOM_BOT_ID"):
            platforms.append("WeCom")
        if get_env_value("WEIXIN_ACCOUNT_ID"):
            platforms.append("Weixin")
        if get_env_value("BLUEBUBBLES_SERVER_URL"):
            platforms.append("BlueBubbles")
        if get_env_value("WEBHOOK_ENABLED"):
            platforms.append("Webhooks")
        if platforms:
            return ", ".join(platforms)
        return None  # No platforms configured — section must run

    elif section_key == "tools":
        tools = []
        if get_env_value("ELEVENLABS_API_KEY"):
            tools.append("TTS/ElevenLabs")
        if get_env_value("BROWSERBASE_API_KEY"):
            tools.append("Browser")
        if get_env_value("FIRECRAWL_API_KEY"):
            tools.append("Firecrawl")
        if tools:
            return ", ".join(tools)
        return None

    return None


def _skip_configured_section(
    config: dict, section_key: str, label: str
) -> bool:
    """Show an already-configured section summary and offer to skip.

    Returns True if the user chose to skip, False if the section should run.
    """
    summary = _get_section_config_summary(config, section_key)
    if not summary:
        return False
    print()
    print_success(f"  {label}: {summary}")
    return not prompt_yes_no(f"  Reconfigure {label.lower()}?", default=False)


# =============================================================================
# OpenClaw Migration
# =============================================================================


_OPENCLAW_SCRIPT = (
    get_optional_skills_dir(PROJECT_ROOT / "optional-skills")
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)


def _load_openclaw_migration_module():
    """Load the openclaw_to_hermes migration script as a module.

    Returns the loaded module, or None if the script can't be loaded.
    """
    if not _OPENCLAW_SCRIPT.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "openclaw_to_hermes", _OPENCLAW_SCRIPT
    )
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve the module
    # (Python 3.11+ requires this for dynamically loaded modules)
    import sys as _sys
    _sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        _sys.modules.pop(spec.name, None)
        raise
    return mod


# Item kinds that represent high-impact changes warranting explicit warnings.
# Gateway tokens/channels can hijack messaging platforms from the old agent.
# Config values may have different semantics between OpenClaw and Hermes.
# Instruction/context files (.md) can contain incompatible setup procedures.
_HIGH_IMPACT_KIND_KEYWORDS = {
    "gateway": "⚠ Gateway/messaging — this will configure Hermes to use your OpenClaw messaging channels",
    "telegram": "⚠ Telegram — this will point Hermes at your OpenClaw Telegram bot",
    "slack": "⚠ Slack — this will point Hermes at your OpenClaw Slack workspace",
    "discord": "⚠ Discord — this will point Hermes at your OpenClaw Discord bot",
    "whatsapp": "⚠ WhatsApp — this will point Hermes at your OpenClaw WhatsApp connection",
    "config": "⚠ Config values — OpenClaw settings may not map 1:1 to Hermes equivalents",
    "soul": "⚠ Instruction file — may contain OpenClaw-specific setup/restart procedures",
    "memory": "⚠ Memory/context file — may reference OpenClaw-specific infrastructure",
    "context": "⚠ Context file — may contain OpenClaw-specific instructions",
}


def _print_migration_preview(report: dict):
    """Print a detailed dry-run preview of what migration would do.

    Groups items by category and adds explicit warnings for high-impact
    changes like gateway token takeover and config value differences.
    """
    items = report.get("items", [])
    if not items:
        print_info("Nothing to migrate.")
        return

    migrated_items = [i for i in items if i.get("status") == "migrated"]
    conflict_items = [i for i in items if i.get("status") == "conflict"]
    skipped_items = [i for i in items if i.get("status") == "skipped"]

    warnings_shown = set()

    if migrated_items:
        print(color("  Would import:", Colors.GREEN))
        for item in migrated_items:
            kind = item.get("kind", "unknown")
            dest = item.get("destination", "")
            if dest:
                dest_short = str(dest).replace(str(Path.home()), "~")
                print(f"      {kind:<22s} → {dest_short}")
            else:
                print(f"      {kind}")

            # Check for high-impact items and collect warnings
            kind_lower = kind.lower()
            dest_lower = str(dest).lower()
            for keyword, warning in _HIGH_IMPACT_KIND_KEYWORDS.items():
                if keyword in kind_lower or keyword in dest_lower:
                    warnings_shown.add(warning)
        print()

    if conflict_items:
        print(color("  Would overwrite (conflicts with existing Hermes config):", Colors.YELLOW))
        for item in conflict_items:
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "already exists")
            print(f"      {kind:<22s}  {reason}")
        print()

    if skipped_items:
        print(color("  Would skip:", Colors.DIM))
        for item in skipped_items:
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "")
            print(f"      {kind:<22s}  {reason}")
        print()

    # Print collected warnings
    if warnings_shown:
        print(color("  ── Warnings ──", Colors.YELLOW))
        for warning in sorted(warnings_shown):
            print(color(f"    {warning}", Colors.YELLOW))
        print()
        print(color("  Note: OpenClaw config values may have different semantics in Hermes.", Colors.YELLOW))
        print(color("  For example, OpenClaw's tool_call_execution: \"auto\" ≠ Hermes's yolo mode.", Colors.YELLOW))
        print(color("  Instruction files (.md) from OpenClaw may contain incompatible procedures.", Colors.YELLOW))
        print()


def _offer_openclaw_migration(hermes_home: Path) -> bool:
    """Detect ~/.openclaw and offer to migrate during first-time setup.

    Runs a dry-run first to show the user exactly what would be imported,
    overwritten, or taken over. Only executes after explicit confirmation.

    Returns True if migration ran successfully, False otherwise.
    """
    openclaw_dir = Path.home() / ".openclaw"
    if not openclaw_dir.is_dir():
        return False

    if not _OPENCLAW_SCRIPT.exists():
        return False

    print()
    print_header("OpenClaw Installation Detected")
    print_info(f"Found OpenClaw data at {openclaw_dir}")
    print_info("Hermes can preview what would be imported before making any changes.")
    print()

    if not prompt_yes_no("Would you like to see what can be imported?", default=True):
        print_info(
            "Skipping migration. You can run it later with: hermes claw migrate --dry-run"
        )
        return False

    # Ensure config.yaml exists before migration tries to read it
    config_path = get_config_path()
    if not config_path.exists():
        save_config(load_config())

    # Load the migration module
    try:
        mod = _load_openclaw_migration_module()
        if mod is None:
            print_warning("Could not load migration script.")
            return False
    except Exception as e:
        print_warning(f"Could not load migration script: {e}")
        logger.debug("OpenClaw migration module load error", exc_info=True)
        return False

    # ── Phase 1: Dry-run preview ──
    try:
        selected = mod.resolve_selected_options(None, None, preset="full")
        dry_migrator = mod.Migrator(
            source_root=openclaw_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=False,  # dry-run — no files modified
            workspace_target=None,
            overwrite=True,  # show everything including conflicts
            migrate_secrets=True,
            output_dir=None,
            selected_options=selected,
            preset_name="full",
        )
        preview_report = dry_migrator.migrate()
    except Exception as e:
        print_warning(f"Migration preview failed: {e}")
        logger.debug("OpenClaw migration preview error", exc_info=True)
        return False

    # Display the full preview
    preview_summary = preview_report.get("summary", {})
    preview_count = preview_summary.get("migrated", 0)

    if preview_count == 0:
        print()
        print_info("Nothing to import from OpenClaw.")
        return False

    print()
    print_header(f"Migration Preview — {preview_count} item(s) would be imported")
    print_info("No changes have been made yet. Review the list below:")
    print()
    _print_migration_preview(preview_report)

    # ── Phase 2: Confirm and execute ──
    if not prompt_yes_no("Proceed with migration?", default=False):
        print_info(
            "Migration cancelled. You can run it later with: hermes claw migrate"
        )
        print_info(
            "Use --dry-run to preview again, or --preset minimal for a lighter import."
        )
        return False

    # Execute the migration — overwrite=False so existing Hermes configs are
    # preserved. The user saw the preview; conflicts are skipped by default.
    try:
        migrator = mod.Migrator(
            source_root=openclaw_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=True,
            workspace_target=None,
            overwrite=False,  # preserve existing Hermes config
            migrate_secrets=True,
            output_dir=None,
            selected_options=selected,
            preset_name="full",
        )
        report = migrator.migrate()
    except Exception as e:
        print_warning(f"Migration failed: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        return False

    # Print final summary
    summary = report.get("summary", {})
    migrated = summary.get("migrated", 0)
    skipped = summary.get("skipped", 0)
    conflicts = summary.get("conflict", 0)
    errors = summary.get("error", 0)

    print()
    if migrated:
        print_success(f"Imported {migrated} item(s) from OpenClaw.")
    if conflicts:
        print_info(f"Skipped {conflicts} item(s) that already exist in Hermes (use hermes claw migrate --overwrite to force).")
    if skipped:
        print_info(f"Skipped {skipped} item(s) (not found or unchanged).")
    if errors:
        print_warning(f"{errors} item(s) had errors — check the migration report.")

    output_dir = report.get("output_dir")
    if output_dir:
        print_info(f"Full report saved to: {output_dir}")

    print_success("Migration complete! Continuing with setup...")
    return True


# =============================================================================
# Main Wizard Orchestrator
# =============================================================================

SETUP_SECTIONS = [
    ("model", "Model & Provider", setup_model_provider),
    ("tts", "Text-to-Speech", setup_tts),
    ("terminal", "Terminal Backend", setup_terminal_backend),
    ("gateway", "Messaging Platforms (Gateway)", setup_gateway),
    ("tools", "Tools", setup_tools),
    ("agent", "Agent Settings", setup_agent_settings),
]

# The returning-user menu intentionally omits standalone TTS because model setup
# already includes TTS selection and tools setup covers the rest of the provider
# configuration. Keep this list in the same order as the visible menu entries.
RETURNING_USER_MENU_SECTION_KEYS = [
    "model",
    "terminal",
    "gateway",
    "tools",
    "agent",
]


def run_setup_wizard(args):
    """Run the interactive setup wizard.

    Supports full, quick, and section-specific setup:
      hermes setup           — full or quick (auto-detected)
      hermes setup model     — just model/provider
      hermes setup tts       — just text-to-speech
      hermes setup terminal  — just terminal backend
      hermes setup gateway   — just messaging platforms
      hermes setup tools     — just tool configuration
      hermes setup agent     — just agent settings
    """
    from hermes_cli.config import is_managed, managed_error
    if is_managed():
        managed_error("run setup wizard")
        return
    ensure_hermes_home()

    reset_requested = bool(getattr(args, "reset", False))
    if reset_requested:
        save_config(copy.deepcopy(DEFAULT_CONFIG))
        print_success("Configuration reset to defaults.")

    config = load_config()
    hermes_home = get_hermes_home()

    # Detect non-interactive environments (headless SSH, Docker, CI/CD)
    non_interactive = getattr(args, 'non_interactive', False)
    if not non_interactive and not is_interactive_stdin():
        non_interactive = True

    if non_interactive:
        print_noninteractive_setup_guidance(
            "Running in a non-interactive environment (no TTY detected)."
        )
        return

    # Check if a specific section was requested
    section = getattr(args, "section", None)
    if section:
        for key, label, func in SETUP_SECTIONS:
            if key == section:
                print()
                print(
                    color(
                        "┌─────────────────────────────────────────────────────────┐",
                        Colors.MAGENTA,
                    )
                )
                print(color(f"│     ⚕ Hermes Setup — {label:<34s} │", Colors.MAGENTA))
                print(
                    color(
                        "└─────────────────────────────────────────────────────────┘",
                        Colors.MAGENTA,
                    )
                )
                func(config)
                save_config(config)
                print()
                print_success(f"{label} configuration complete!")
                return

        print_error(f"Unknown setup section: {section}")
        print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return

    # Check if this is an existing installation with a provider configured
    from hermes_cli.auth import get_active_provider

    active_provider = get_active_provider()
    is_existing = (
        bool(get_env_value("OPENROUTER_API_KEY"))
        or bool(get_env_value("OPENAI_BASE_URL"))
        or active_provider is not None
    )

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│             ⚕ Hermes Agent Setup Wizard                │", Colors.MAGENTA
        )
    )
    print(
        color(
            "├─────────────────────────────────────────────────────────┤",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│  Hermes Agent 설치를 함께 설정해볼게요.            │", Colors.MAGENTA
        )
    )
    print(
        color(
            "│  언제든 Ctrl+C를 눌러 종료할 수 있습니다.         │", Colors.MAGENTA
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )

    migration_ran = False

    if is_existing:
        # ── Returning User Menu ──
        print()
        print_header("다시 오신 것을 환영합니다!")
        print_success("이미 Hermes가 설정되어 있습니다.")
        print()

        menu_choices = [
            "빠른 설정 - 누락된 항목만 설정",
            "전체 설정 - 모든 항목 다시 설정",
            "모델 및 Provider",
            "터미널 백엔드",
            "메시징 플랫폼 (Gateway)",
            "도구",
            "에이전트 설정",
            "종료",
        ]
        choice = prompt_choice("무엇을 하시겠어요?", menu_choices, 0)

        if choice == 0:
            # Quick setup
            _run_quick_setup(config, hermes_home)
            return
        elif choice == 1:
            # Full setup — fall through to run all sections
            pass
        elif choice == 7:
            print_info("종료합니다. 준비되면 다시 'hermes setup'을 실행하세요.")
            return
        elif 2 <= choice <= 6:
            # Individual section — map by key, not by position.
            # SETUP_SECTIONS includes TTS but the returning-user menu skips it,
            # so positional indexing (choice - 2) would dispatch the wrong section.
            section_key = RETURNING_USER_MENU_SECTION_KEYS[choice - 2]
            section = next((s for s in SETUP_SECTIONS if s[0] == section_key), None)
            if section:
                _, label, func = section
                func(config)
                save_config(config)
                _print_setup_summary(config, hermes_home)
            return
    else:
        # ── First-Time Setup ──
        print()

        # Offer OpenClaw migration before configuration begins
        migration_ran = _offer_openclaw_migration(hermes_home)
        if migration_ran:
            config = load_config()

        setup_mode = prompt_choice("How would you like to set up Hermes?", [
            "Quick setup — provider, model & messaging (recommended)",
            "Full setup — configure everything",
        ], 0)

        if setup_mode == 0:
            _run_first_time_quick_setup(config, hermes_home, is_existing)
            return

    # ── Full Setup — run all sections ──
    print_header("설정 위치")
    print_info(f"Config 파일:   {get_config_path()}")
    print_info(f"Secrets 파일:  {get_env_path()}")
    print_info(f"데이터 폴더:   {hermes_home}")
    print_info(f"설치 디렉터리: {PROJECT_ROOT}")
    print()
    print_info("이 파일들은 직접 수정하거나 'hermes config edit'를 사용할 수 있습니다")

    if migration_ran:
        print()
        print_info("설정이 OpenClaw에서 가져와졌습니다.")
        print_info("아래 각 섹션에는 가져온 내용이 표시되며, Enter를 누르면 유지되고")
        print_info("필요하면 다시 설정할 수 있습니다.")

    # Section 1: Model & Provider
    if not (migration_ran and _skip_configured_section(config, "model", "Model & Provider")):
        setup_model_provider(config)

    # Section 2: Terminal Backend
    if not (migration_ran and _skip_configured_section(config, "terminal", "Terminal Backend")):
        setup_terminal_backend(config)

    # Section 3: Agent Settings
    if not (migration_ran and _skip_configured_section(config, "agent", "Agent Settings")):
        setup_agent_settings(config)

    # Section 4: Messaging Platforms
    if not (migration_ran and _skip_configured_section(config, "gateway", "Messaging Platforms")):
        setup_gateway(config)

    # Section 5: Tools
    if not (migration_ran and _skip_configured_section(config, "tools", "Tools")):
        setup_tools(config, first_install=not is_existing)

    # Save and show summary
    save_config(config)
    _print_setup_summary(config, hermes_home)

    _offer_launch_chat()


def _resolve_hermes_chat_argv() -> Optional[list[str]]:
    """Resolve argv for launching ``hermes chat`` in a fresh process."""
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin, "chat"]

    try:
        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main", "chat"]
    except Exception:
        pass

    return None


def _offer_launch_chat():
    """Prompt the user to jump straight into chat after setup."""
    print()
    if not prompt_yes_no("지금 대화를 시작할까요?", True):
        return

    chat_argv = _resolve_hermes_chat_argv()
    if not chat_argv:
        print_info("자동으로 Hermes를 다시 실행할 수 없습니다. 'hermes chat'을 수동으로 실행하세요.")
        return

    os.execvp(chat_argv[0], chat_argv)


def _run_first_time_quick_setup(config: dict, hermes_home, is_existing: bool):
    """Streamlined first-time setup: provider + model only.

    Applies sensible defaults for TTS (Edge), terminal (local), agent
    settings, and tools — the user can customize later via
    ``hermes setup <section>``.
    """
    # Step 1: Model & Provider (essential — skips rotation/vision/TTS)
    setup_model_provider(config, quick=True)

    # Step 2: Apply defaults for everything else
    _apply_default_agent_settings(config)
    config.setdefault("terminal", {}).setdefault("backend", "local")

    save_config(config)

    # Step 3: Offer messaging gateway setup
    print()
    gateway_choice = prompt_choice(
        "메시징 플랫폼을 연결할까요? (Telegram, Discord 등)",
        [
            "지금 메시징 설정하기 (권장)",
            "건너뛰기 — 나중에 'hermes setup gateway'로 설정",
        ],
        0,
    )

    if gateway_choice == 0:
        setup_gateway(config)
        save_config(config)

    print()
    print_success("설정이 완료되었습니다! 바로 사용할 수 있어요.")
    print()
    print_info("  모든 설정 구성:        hermes setup")
    if gateway_choice != 0:
        print_info("  Telegram/Discord 연결:  hermes setup gateway")
    print()

    _print_setup_summary(config, hermes_home)

    _offer_launch_chat()


def _run_quick_setup(config: dict, hermes_home):
    """Quick setup — only configure items that are missing."""
    from hermes_cli.config import (
        get_missing_env_vars,
        get_missing_config_fields,
        check_config_version,
    )

    print()
    print_header("빠른 설정 — 누락된 항목만")

    # Check what's missing
    missing_required = [
        v for v in get_missing_env_vars(required_only=False) if v.get("is_required")
    ]
    missing_optional = [
        v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")
    ]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()

    has_anything_missing = (
        missing_required
        or missing_optional
        or missing_config
        or current_ver < latest_ver
    )

    if not has_anything_missing:
        print_success("모든 항목이 설정되어 있습니다! 할 일이 없습니다.")
        print()
        print_info("다시 설정하려면 'hermes setup'을 실행하고 '전체 설정'을 선택하세요,")
        print_info("또는 메뉴에서 특정 섹션을 선택하세요.")
        return

    # Handle missing required env vars
    if missing_required:
        print()
        print_info(f"{len(missing_required)}개의 필수 설정이 누락되었습니다:")
        for var in missing_required:
            print(f"     • {var['name']}")
        print()

        for var in missing_required:
            print()
            print(color(f"  {var['name']}", Colors.CYAN))
            print_info(f"  {var.get('description', '')}")
            if var.get("url"):
                print_info(f"  키 발급 위치: {var['url']}")

            if var.get("password"):
                value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
            else:
                value = prompt(f"  {var.get('prompt', var['name'])}")

            if value:
                save_env_value(var["name"], value)
                print_success(f"  Saved {var['name']}")
            else:
                print_warning(f"  Skipped {var['name']}")

    # Split missing optional vars by category
    missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
    missing_messaging = [
        v
        for v in missing_optional
        if v.get("category") == "messaging" and not v.get("advanced")
    ]

    # ── Tool API keys (checklist) ──
    if missing_tools:
        print()
        print_header("도구 API 키")

        checklist_labels = []
        for var in missing_tools:
            tools = var.get("tools", [])
            tools_str = f" → {', '.join(tools[:2])}" if tools else ""
            checklist_labels.append(f"{var.get('description', var['name'])}{tools_str}")

        selected_indices = prompt_checklist(
            "어떤 도구를 설정하시겠어요?",
            checklist_labels,
        )

        for idx in selected_indices:
            var = missing_tools[idx]
            _prompt_api_key(var)

    # ── Messaging platforms (checklist then prompt for selected) ──
    if missing_messaging:
        print()
        print_header("메시징 플랫폼")
        print_info("어디서든 Hermes와 대화할 수 있도록 메시징 앱에 연결하세요.")
        print_info("이 설정은 나중에 'hermes setup gateway'로 다시 할 수 있습니다.")

        # Group by platform (preserving order)
        platform_order = []
        platforms = {}
        for var in missing_messaging:
            name = var["name"]
            if "TELEGRAM" in name:
                plat = "Telegram"
            elif "DISCORD" in name:
                plat = "Discord"
            elif "SLACK" in name:
                plat = "Slack"
            else:
                continue
            if plat not in platforms:
                platform_order.append(plat)
            platforms.setdefault(plat, []).append(var)

        platform_labels = [
            {
                "Telegram": "📱 Telegram",
                "Discord": "💬 Discord",
                "Slack": "💼 Slack",
            }.get(p, p)
            for p in platform_order
        ]

        selected_indices = prompt_checklist(
            "어떤 플랫폼을 설정하시겠어요?",
            platform_labels,
        )

        for idx in selected_indices:
            plat = platform_order[idx]
            vars_list = platforms[plat]
            emoji = {"Telegram": "📱", "Discord": "💬", "Slack": "💼"}.get(plat, "")
            print()
            print(color(f"  ─── {emoji} {plat} ───", Colors.CYAN))
            print()
            for var in vars_list:
                print_info(f"  {var.get('description', '')}")
                if var.get("url"):
                    print_info(f"  {var['url']}")
                if var.get("password"):
                    value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
                else:
                    value = prompt(f"  {var.get('prompt', var['name'])}")
                if value:
                    save_env_value(var["name"], value)
                    print_success("  ✓ Saved")
                else:
                    print_warning("  Skipped")
                print()

    # Handle missing config fields
    if missing_config:
        print()
        print_info(
            f"Adding {len(missing_config)} new config option(s) with defaults..."
        )
        for field in missing_config:
            print_success(f"  Added {field['key']} = {field['default']}")

        # Update config version
        config["_config_version"] = latest_ver
        save_config(config)

    # Jump to summary
    _print_setup_summary(config, hermes_home)

"""Text-to-speech provider setup (provider picker, API-key prompts, local engine installs, xAI OAuth).
setup.py names are resolved through the module object so test patches on ``hermes_cli.setup.<name>``
take effect; setup.py re-exports the public entry points."""

import logging
import shutil
import subprocess
import sys
from tools import tool_backend_helpers
from hermes_cli import nous_subscription

logger = logging.getLogger("hermes_cli.setup")


def _pip_install_tts_package(name: str, pip_args: list, manual_cmd: str) -> bool:
    """Install a local TTS engine through the canonical uv → pip → ensurepip
    ladder so pip-less venvs (Ubuntu 25.10 ``python -m venv``, ``uv venv``) work."""
    from hermes_cli.tools_config import _pip_install
    try:
        result = _pip_install(pip_args, timeout=300)
        if result.returncode == 0:
            _setup.print_success(f"{name} installed successfully")
            return True
        err = (result.stderr or "").strip()
        reason = err[:300] if err else "install failed"
    except Exception as e:
        reason = e
    _setup.print_error(f"Failed to install {name}: {reason}")
    _setup.print_info(f"Try manually: {manual_cmd}")
    return False


# sys.platform -> (manual install hint, install command); anything else uses "linux".
_ESPEAK_INSTALL = {
    "darwin": ("Install with: brew install espeak-ng", ["brew", "install", "espeak-ng"]),
    "win32": ("Install with: choco install espeak-ng", ["choco", "install", "espeak-ng", "-y"]),
    "linux": ("Install with: sudo apt install espeak-ng", ["sudo", "apt", "install", "-y", "espeak-ng"]),
}


def _install_neutts_deps() -> bool:
    """Install NeuTTS dependencies with user approval. Returns True on success."""
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        hint, install_cmd = _ESPEAK_INSTALL.get(sys.platform, _ESPEAK_INSTALL["linux"])
        print()
        _setup.print_warning("NeuTTS requires espeak-ng for phonemization.")
        _setup.print_info(hint)
        print()
        if _setup.prompt_yes_no("Install espeak-ng now?", True):
            try:
                subprocess.run(install_cmd, check=True)
                _setup.print_success("espeak-ng installed")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                _setup.print_warning(f"Could not install espeak-ng automatically: {e}")
                _setup.print_info("Please install it manually and re-run setup.")
                return False
        else:
            _setup.print_warning("espeak-ng is required for NeuTTS. Install it manually before using NeuTTS.")

    _setup._info(None, "Installing neutts Python package...",
          "This will also download the TTS model (~300MB) on first use.", None)
    return _pip_install_tts_package("neutts", ["-U", "neutts[all]", "--quiet"], "uv pip install -U 'neutts[all]'")


def _install_kittentts_deps() -> bool:
    """Install KittenTTS dependencies with user approval. Returns True on success."""
    wheel_url = "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"
    _setup._info(None, "Installing kittentts Python package (~25-80MB model downloaded on first use)...", None)
    return _pip_install_tts_package(
        "kittentts", ["-U", wheel_url, "soundfile", "--quiet"], f"uv pip install -U '{wheel_url}' soundfile")


def _xai_oauth_logged_in_for_setup() -> bool:
    """True iff xAI Grok OAuth credentials are stored locally, so TTS/STT setup can skip the
    API-key prompt for users who logged in via ``hermes model`` -> xAI Grok OAuth."""
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        return bool(get_xai_oauth_auth_status().get("logged_in"))
    except Exception:
        return False


def _run_xai_oauth_login_from_setup() -> bool:
    """Run the xAI Grok OAuth device-code login from inside the setup wizard. Saves OAuth tokens
    only — does **not** switch the active provider or rewrite ``model.provider`` (callers only need
    credentials for side tools). False on any failure (caller falls back)."""
    try:
        from hermes_cli.auth import (
            _is_remote_session, _save_xai_oauth_tokens, _xai_oauth_device_code_login,
            unsuppress_credential_source)
    except Exception as exc:
        _setup.print_warning(f"xAI Grok OAuth helpers unavailable: {exc}")
        return False
    _setup._info(None, "Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    try:
        creds = _xai_oauth_device_code_login(open_browser=not _is_remote_session())
        _save_xai_oauth_tokens(
            creds["tokens"], discovery=creds.get("discovery"), redirect_uri=creds.get("redirect_uri", ""),
            last_refresh=creds.get("last_refresh"), auth_mode="oauth_device_code", set_active=False)
        # Mirror model/dashboard re-login: clear device_code suppression so the pool can seed
        # from the singleton after a prior `auth remove`.
        unsuppress_credential_source("xai-oauth", "device_code")
        return True
    except Exception as exc:
        _setup.print_warning(f"xAI Grok OAuth login failed: {exc}")
        return False


_TTS_PROVIDER_CHOICES = [
    ("edge", "Edge TTS (free, cloud-based, no setup needed)"),
    ("elevenlabs", "ElevenLabs (premium quality, needs API key)"),
    ("openai", "OpenAI TTS (good quality, needs API key)"),
    ("xai", "xAI TTS (Grok voices — OAuth login or API key)"),
    ("minimax", "MiniMax TTS (high quality with voice cloning, needs API key)"),
    ("mistral", "Mistral Voxtral TTS (multilingual, native Opus, needs API key)"),
    ("gemini", "Google Gemini TTS (30 prebuilt voices, prompt-controllable, needs API key)"),
    ("neutts", "NeuTTS (local on-device, free, ~300MB model download)"),
    ("kittentts", "KittenTTS (local on-device, free, lightweight ~25-80MB ONNX)")]
# Short label = menu label minus its parenthetical ("Edge TTS", "Mistral Voxtral TTS", ...).
_TTS_PROVIDER_LABELS = {key: label.split(" (")[0] for key, label in _TTS_PROVIDER_CHOICES}
# provider -> (env vars that satisfy it, env var to save, prompt, success line, pre-prompt hint)
_TTS_API_KEY_PROVIDERS = {
    "elevenlabs": (("ELEVENLABS_API_KEY",), "ELEVENLABS_API_KEY", "ElevenLabs API key",
                   "ElevenLabs API key saved", ""),
    "openai": (("VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"), "VOICE_TOOLS_OPENAI_KEY",
               "OpenAI API key for TTS", "OpenAI TTS API key saved", ""),
    "minimax": (("MINIMAX_API_KEY",), "MINIMAX_API_KEY", "MiniMax API key for TTS",
                "MiniMax TTS API key saved", ""),
    "mistral": (("MISTRAL_API_KEY",), "MISTRAL_API_KEY", "Mistral API key for TTS",
                "Mistral TTS API key saved", ""),
    "gemini": (("GEMINI_API_KEY", "GOOGLE_API_KEY"), "GEMINI_API_KEY", "Gemini API key for TTS",
               "Gemini TTS API key saved", "Get a free API key at https://aistudio.google.com/app/apikey"),
}
# provider -> (module, display name, requirement lines, install question, installer)
_TTS_LOCAL_PROVIDERS = {
    "neutts": ("neutts", "NeuTTS",
               ("NeuTTS requires:", "  • Python package: neutts (~50MB install + ~300MB model on first use)",
                "  • System package: espeak-ng (phonemizer)"),
               "Install NeuTTS dependencies now?", _install_neutts_deps),
    "kittentts": ("kittentts", "KittenTTS",
                  ("KittenTTS is lightweight (~25-80MB, CPU-only, no API key required).",
                   "Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo"),
                  "Install KittenTTS now?", _install_kittentts_deps)}


def _tts_api_key_step(selected: str) -> str:
    """Ensure the key for an API-key TTS provider exists; fall back to edge otherwise."""
    env_vars, save_var, prompt_label, saved_msg, hint = _TTS_API_KEY_PROVIDERS[selected]
    if any(_setup.get_env_value(v) for v in env_vars):
        return selected
    print()
    if hint:
        _setup.print_info(hint)
    api_key = _setup.prompt(prompt_label, password=True)
    if api_key:
        _setup.save_env_value(save_var, api_key)
        _setup.print_success(saved_msg)
        return selected
    _setup.print_warning("No API key provided. Falling back to Edge TTS.")
    return "edge"


def _tts_local_install_step(selected: str) -> str:
    """Offer to install a local TTS engine; fall back to edge if declined/failed."""
    module, name, lines, question, installer = _TTS_LOCAL_PROVIDERS[selected]
    if _setup._module_installed(module):
        _setup.print_success(f"{name} is already installed")
        return selected
    print()
    for line in lines:
        _setup.print_info(line)
    print()
    if not _setup.prompt_yes_no(question, True):
        _setup.print_info(f"Skipping install. Set tts.provider to '{selected}' after installing manually.")
        return "edge"
    if not installer():
        _setup.print_warning(f"{name} installation incomplete. Falling back to Edge TTS.")
        return "edge"
    return selected


def _xai_oauth_path():
    if _run_xai_oauth_login_from_setup():
        _setup.print_success("Logged in — xAI TTS will use these OAuth credentials")
        return None
    return "xAI Grok OAuth login did not complete. Falling back to Edge TTS."


def _xai_api_key_path():
    api_key = _setup.prompt("xAI API key for TTS", password=True)
    if api_key:
        _setup.save_env_value("XAI_API_KEY", api_key)
        _setup.print_success("xAI TTS API key saved")
        return None
    from hermes_constants import display_hermes_home as _dhh
    return ("No xAI API key provided for TTS. Configure XAI_API_KEY via hermes setup model "
            f"or {_dhh()}/.env to use xAI TTS. Falling back to Edge TTS.")


def _tts_xai_step(config: dict) -> str:
    """xAI TTS auth. Order: existing OAuth tokens (free for SuperGrok) > existing
    XAI_API_KEY > offer both paths — xAI TTS works with OAuth bearer tokens too."""
    if _xai_oauth_logged_in_for_setup():
        _setup.print_success("xAI TTS will use your xAI Grok OAuth (SuperGrok / Premium+) credentials")
    elif _setup.get_env_value("XAI_API_KEY"):
        _setup.print_success("xAI TTS will use your existing XAI_API_KEY")
    else:
        print()
        choice_idx = _setup.prompt_choice(
            "How do you want xAI TTS to authenticate?",
            choices=["Sign in with xAI Grok OAuth (SuperGrok / Premium+) — browser login",
                     "Paste an xAI API key (console.x.ai)", "Skip → fallback to Edge TTS"], default=0)
        # Each path returns the fallback warning (result is then "edge") or None on success.
        fallback = (_xai_oauth_path, _xai_api_key_path, lambda: "xAI TTS skipped. Falling back to Edge TTS.")[
            choice_idx if choice_idx in (0, 1) else 2]()
        if fallback:
            _setup.print_warning(fallback)
            return "edge"
    print()
    voice_id = (_setup.prompt("xAI voice_id (Enter for 'eve', or paste a custom voice ID)") or "").strip()
    if voice_id:
        config.setdefault("tts", {}).setdefault("xai", {})["voice_id"] = voice_id
        _setup.print_success(f"xAI voice_id set to: {voice_id}")
    return "xai"


def _setup_tts_provider(config: dict):
    """Interactive TTS provider selection with install flow for local engines."""
    current_provider = config.get("tts", {}).get("provider", "edge")
    current_label = _TTS_PROVIDER_LABELS.get(current_provider, current_provider)
    print()
    _setup.print_header("Text-to-Speech Provider (optional)")
    _setup._info(f"Current: {current_label}", None)
    options = list(_TTS_PROVIDER_CHOICES)
    if tool_backend_helpers.managed_nous_tools_enabled() and nous_subscription.get_nous_subscription_features(config).nous_auth_present:
        options.insert(0, ("nous-openai",
                           "Nous Subscription (managed OpenAI TTS, billed to your subscription)"))
    choices = [label for _, label in options] + [f"Keep current ({current_label})"]
    keep_current_idx = len(choices) - 1
    idx = _setup.prompt_choice("Select TTS provider:", choices, keep_current_idx)
    if idx == keep_current_idx:
        return
    selected = options[idx][0]
    if selected == "nous-openai":
        selected = "openai"
        _setup.print_info("OpenAI TTS will use the managed Nous gateway and bill to your subscription.")
        if _setup.get_env_value("VOICE_TOOLS_OPENAI_KEY") or _setup.get_env_value("OPENAI_API_KEY"):
            _setup.print_warning("Direct OpenAI credentials are still configured and may take precedence "
                                 "until removed from ~/.hermes/.env.")
    elif selected in _TTS_LOCAL_PROVIDERS:
        selected = _tts_local_install_step(selected)
    elif selected in _TTS_API_KEY_PROVIDERS:
        selected = _tts_api_key_step(selected)
    elif selected == "xai":
        selected = _tts_xai_step(config)
    config.setdefault("tts", {})["provider"] = selected
    _setup.save_config(config)
    _setup.print_success(f"TTS provider set to: {_TTS_PROVIDER_LABELS.get(selected, selected)}")


def setup_tts(config: dict):
    """Standalone TTS setup (for 'hermes setup tts')."""
    _setup_tts_provider(config)


import hermes_cli.setup as _setup  # noqa: E402  (bottom: hermes_cli.setup imports this module)

"""Shared fixtures for the hermes-agent test suite.

Hermetic-test invariants enforced here (see AGENTS.md for rationale):

1. **No credential env vars.** All provider/credential-shaped env vars
   (ending in _API_KEY, _TOKEN, _SECRET, _PASSWORD, _CREDENTIALS, etc.)
   are unset before every test. Local developer keys cannot leak in.
2. **Isolated HERMES_HOME.** HERMES_HOME points to a per-test tempdir so
   code reading ``~/.hermes/*`` via ``get_hermes_home()`` can't see the
   real one. (We do NOT also redirect HOME — that broke subprocesses in
   CI. Code using ``Path.home() / ".hermes"`` instead of the canonical
   ``get_hermes_home()`` is a bug to fix at the callsite.)
3. **Deterministic runtime.** TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0.
4. **No HERMES_SESSION_* inheritance** — the agent's current gateway
   session must not leak into tests.

These invariants make the local test run match CI closely. Gaps that
remain (CPU count, xdist worker count) are addressed by the canonical
test runner at ``scripts/run_tests.sh``.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Per-file process isolation ──────────────────────────────────────────────
# Tests run via ``scripts/run_tests_parallel.py``, which spawns a fresh
# ``python -m pytest <file>`` subprocess per test file. Cross-file state
# leakage (module-level dicts, ContextVars, caches) is impossible: each
# file gets a clean Python interpreter. Intra-file ordering is the test
# author's responsibility — if test A in foo.py mutates state that test B
# in foo.py reads, that's a real bug to fix in the file (it would also
# bite anyone running ``pytest tests/foo.py`` directly).
#
# This replaces the historic _reset_module_state autouse fixture (manual
# state clearing) and the brief experiment with subprocess-per-test
# isolation (too slow at ~17k tests).
#
# See ``scripts/run_tests_parallel.py`` for the runner.


# ── Credential env-var filter ──────────────────────────────────────────────
#
# Any env var in the current process matching ONE of these patterns is
# unset for every test. Developers' local keys cannot leak into assertions
# about "auto-detect provider when key present".

_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIALS",
    "_ACCESS_KEY",
    "_SECRET_ACCESS_KEY",
    "_PRIVATE_KEY",
    "_OAUTH_TOKEN",
    "_WEBHOOK_SECRET",
    "_ENCRYPT_KEY",
    "_APP_SECRET",
    "_CLIENT_SECRET",
    "_CORP_SECRET",
    "_AES_KEY",
)

# Explicit names (for ones that don't fit the suffix pattern)
_CREDENTIAL_NAMES = frozenset({
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_TOKEN",
    "FAL_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "NOUS_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "MINIMAX_API_KEY",
    "OLLAMA_API_KEY",
    "OPENVIKING_API_KEY",
    "COPILOT_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "BROWSERBASE_API_KEY",
    "FIRECRAWL_API_KEY",
    "PARALLEL_API_KEY",
    "EXA_API_KEY",
    "TAVILY_API_KEY",
    "WANDB_API_KEY",
    "ELEVENLABS_API_KEY",
    "HONCHO_API_KEY",
    "MEM0_API_KEY",
    "SUPERMEMORY_API_KEY",
    "RETAINDB_API_KEY",
    "HINDSIGHT_API_KEY",
    "HINDSIGHT_LLM_API_KEY",
    "DAYTONA_API_KEY",
    "TWILIO_AUTH_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "MATTERMOST_TOKEN",
    "MATRIX_ACCESS_TOKEN",
    "MATRIX_PASSWORD",
    "MATRIX_RECOVERY_KEY",
    "HASS_TOKEN",
    "EMAIL_PASSWORD",
    "BLUEBUBBLES_PASSWORD",
    "FEISHU_APP_SECRET",
    "FEISHU_ENCRYPT_KEY",
    "FEISHU_VERIFICATION_TOKEN",
    "DINGTALK_CLIENT_SECRET",
    "QQ_CLIENT_SECRET",
    "QQ_STT_API_KEY",
    "WECOM_SECRET",
    "WECOM_CALLBACK_CORP_SECRET",
    "WECOM_CALLBACK_TOKEN",
    "WECOM_CALLBACK_ENCODING_AES_KEY",
    "WEIXIN_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "TERMINAL_SSH_KEY",
    "SUDO_PASSWORD",
    "GATEWAY_PROXY_KEY",
    "API_SERVER_KEY",
    "TOOL_GATEWAY_USER_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "WEBHOOK_SECRET",
    "VOICE_TOOLS_OPENAI_KEY",
    "BROWSER_USE_API_KEY",
    "CUSTOM_API_KEY",
    "GATEWAY_PROXY_URL",
    "GEMINI_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OLLAMA_BASE_URL",
    "GROQ_BASE_URL",
    "XAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
})


def _looks_like_credential(name: str) -> bool:
    """True if env var name matches a credential-shaped pattern."""
    if name in _CREDENTIAL_NAMES:
        return True
    return any(name.endswith(suf) for suf in _CREDENTIAL_SUFFIXES)


# HERMES_* vars that change test behavior by being set. Unset all of these
# unconditionally — individual tests that need them set do so explicitly.
_HERMES_BEHAVIORAL_VARS = frozenset({
    "HERMES_YOLO_MODE",
    "HERMES_INTERACTIVE",
    "HERMES_QUIET",
    "HERMES_TOOL_PROGRESS",
    "HERMES_TOOL_PROGRESS_MODE",
    "HERMES_MAX_ITERATIONS",
    "HERMES_SESSION_PLATFORM",
    "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_NAME",
    "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_SOURCE",
    "HERMES_SESSION_KEY",
    "HERMES_GATEWAY_SESSION",
    "HERMES_CRON_SESSION",
    "_HERMES_GATEWAY",
    "HERMES_PLATFORM",
    "HERMES_MODEL",
    "HERMES_INFERENCE_MODEL",
    "HERMES_INFERENCE_PROVIDER",
    "HERMES_TUI_PROVIDER",
    "HERMES_MANAGED",
    "HERMES_MANAGED_DIR",
    "HERMES_DEV",
    "HERMES_CONTAINER",
    "HERMES_EPHEMERAL_SYSTEM_PROMPT",
    "HERMES_TIMEZONE",
    "HERMES_REDACT_SECRETS",
    "HERMES_BACKGROUND_NOTIFICATIONS",
    "HERMES_EXEC_ASK",
    "HERMES_HOME_MODE",
    "HERMES_AGENT_USE_LEGACY_SESSION_KEYS",
    # Kanban path/board pins must never leak from a developer shell or
    # dispatched worker into tests; otherwise tests can write fake tasks to
    # the real ~/.hermes/kanban.db instead of the per-test HERMES_HOME.
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_LOGS_ROOT",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DISPATCH_IN_GATEWAY",
    "HERMES_TENANT",
    # Honcho host selection changes which nested config block wins. A local
    # shell override leaked "myhost" into the full suite and flipped 20
    # otherwise-unrelated config tests away from the default "hermes" host.
    "HERMES_HONCHO_HOST",
    # Dashboard OAuth auth gate (PR #30156). When set, the bundled
    # dashboard-auth `nous` plugin auto-registers itself on plugin discovery,
    # which is triggered by any `/api/status` call. That leaks a provider
    # into the dashboard_auth registry across tests in the same worker and
    # makes assertions like `auth_providers == []` flaky. CI never sets
    # these, so production tests must not see them either.
    "HERMES_DASHBOARD_OAUTH_CLIENT_ID",
    "HERMES_DASHBOARD_PORTAL_URL",
    "TERMINAL_CWD",
    "TERMINAL_ENV",
    "TERMINAL_CONTAINER_CPU",
    "TERMINAL_CONTAINER_DISK",
    "TERMINAL_CONTAINER_MEMORY",
    "TERMINAL_CONTAINER_PERSISTENT",
    "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
    "TERMINAL_DOCKER_ORPHAN_REAPER",
    "TERMINAL_DOCKER_RUN_AS_HOST_USER",
    "BROWSER_CDP_URL",
    "CAMOFOX_URL",
    # Platform allowlists — not credentials, but if set from any source
    # (user shell, earlier leaky test, CI env), they change gateway auth
    # behavior and flake button-authorization tests.
    "TELEGRAM_ALLOWED_USERS",
    "DISCORD_ALLOWED_USERS",
    "WHATSAPP_ALLOWED_USERS",
    "SLACK_ALLOWED_USERS",
    "SIGNAL_ALLOWED_USERS",
    "SIGNAL_GROUP_ALLOWED_USERS",
    "EMAIL_ALLOWED_USERS",
    "SMS_ALLOWED_USERS",
    "MATTERMOST_ALLOWED_USERS",
    "MATRIX_ALLOWED_USERS",
    "DINGTALK_ALLOWED_USERS",
    "FEISHU_ALLOWED_USERS",
    "WECOM_ALLOWED_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    "TELEGRAM_ALLOW_ALL_USERS",
    "DISCORD_ALLOW_ALL_USERS",
    "WHATSAPP_ALLOW_ALL_USERS",
    "SLACK_ALLOW_ALL_USERS",
    "SIGNAL_ALLOW_ALL_USERS",
    "EMAIL_ALLOW_ALL_USERS",
    "SMS_ALLOW_ALL_USERS",
    # Gateway home channels are set by /sethome in real profiles. Tests that
    # exercise dashboard notification toggles must opt in explicitly or they
    # can accidentally subscribe against a developer's real home channel.
    "TELEGRAM_HOME_CHANNEL",
    "TELEGRAM_HOME_CHANNEL_THREAD_ID",
    "TELEGRAM_HOME_CHANNEL_NAME",
    "TELEGRAM_CRON_THREAD_ID",
    "DISCORD_HOME_CHANNEL",
    "DISCORD_HOME_CHANNEL_THREAD_ID",
    "DISCORD_HOME_CHANNEL_NAME",
    "SLACK_HOME_CHANNEL",
    "SLACK_HOME_CHANNEL_THREAD_ID",
    "SLACK_HOME_CHANNEL_NAME",
    "WHATSAPP_HOME_CHANNEL",
    "WHATSAPP_HOME_CHANNEL_THREAD_ID",
    "WHATSAPP_HOME_CHANNEL_NAME",
    "SIGNAL_HOME_CHANNEL",
    "SIGNAL_HOME_CHANNEL_THREAD_ID",
    "SIGNAL_HOME_CHANNEL_NAME",
    "EMAIL_HOME_CHANNEL",
    "EMAIL_HOME_CHANNEL_THREAD_ID",
    "EMAIL_HOME_CHANNEL_NAME",
    "SMS_HOME_CHANNEL",
    "SMS_HOME_CHANNEL_THREAD_ID",
    "SMS_HOME_CHANNEL_NAME",
    "MATTERMOST_HOME_CHANNEL",
    "MATTERMOST_HOME_CHANNEL_THREAD_ID",
    "MATTERMOST_HOME_CHANNEL_NAME",
    "MATRIX_HOME_CHANNEL",
    "MATRIX_HOME_CHANNEL_THREAD_ID",
    "MATRIX_HOME_CHANNEL_NAME",
    "DINGTALK_HOME_CHANNEL",
    "DINGTALK_HOME_CHANNEL_THREAD_ID",
    "DINGTALK_HOME_CHANNEL_NAME",
    "FEISHU_HOME_CHANNEL",
    "FEISHU_HOME_CHANNEL_THREAD_ID",
    "FEISHU_HOME_CHANNEL_NAME",
    "WECOM_HOME_CHANNEL",
    "WECOM_HOME_CHANNEL_THREAD_ID",
    "WECOM_HOME_CHANNEL_NAME",
    # API server bind/auth settings are common in local gateway profiles and
    # change adapter defaults plus load_gateway_config() enablement. Tests that
    # need them set opt in explicitly with monkeypatch.
    "API_SERVER_ENABLED",
    "API_SERVER_HOST",
    "API_SERVER_PORT",
    "API_SERVER_KEY",
    "API_SERVER_CORS_ORIGINS",
    "API_SERVER_MODEL_NAME",
    # Platform gating — set by load_gateway_config() as a side effect when
    # a config.yaml is present, so individual test bodies that call the
    # loader leak these values into later tests in the same process.
    # Force-clear on every test setup so the leak can't happen.
    "SLACK_REQUIRE_MENTION",
    "SLACK_STRICT_MENTION",
    "SLACK_FREE_RESPONSE_CHANNELS",
    "SLACK_ALLOW_BOTS",
    "SLACK_REACTIONS",
    "DISCORD_REQUIRE_MENTION",
    "DISCORD_FREE_RESPONSE_CHANNELS",
    "TELEGRAM_REQUIRE_MENTION",
    "WHATSAPP_REQUIRE_MENTION",
    "DINGTALK_REQUIRE_MENTION",
    "MATRIX_REQUIRE_MENTION",
})


@pytest.fixture(autouse=True)
def _hermetic_environment(tmp_path, monkeypatch):
    """Blank out all credential/behavioral env vars so local and CI match.

    Also redirects HOME and HERMES_HOME to per-test tempdirs so code that
    reads ``~/.hermes/*`` can't touch the real one, and pins TZ/LANG so
    datetime/locale-sensitive tests are deterministic.
    """
    # 1. Blank every credential-shaped env var that's currently set.
    for name in list(os.environ.keys()):
        if _looks_like_credential(name):
            monkeypatch.delenv(name, raising=False)

    # 2. Blank behavioral HERMES_* vars that could change test semantics.
    for name in _HERMES_BEHAVIORAL_VARS:
        monkeypatch.delenv(name, raising=False)

    # Honcho's fallback host/config resolution legitimately reads the user's
    # global ~/.honcho/config.json. Keep HOME stable (subprocess tests depend
    # on it), but pin the host so ordinary tests cannot inherit a developer's
    # defaultHost and silently select the wrong nested config block. Tests of
    # custom host resolution override/delete this explicitly.
    monkeypatch.setenv("HERMES_HONCHO_HOST", "hermes")

    # 3. Redirect HERMES_HOME to a per-test tempdir. Code that reads
    #    ``~/.hermes/*`` via ``get_hermes_home()`` now gets the tempdir.
    #
    #    NOTE: We do NOT also redirect HOME. Doing so broke CI because
    #    some tests (and their transitive deps) spawn subprocesses that
    #    inherit HOME and expect it to be stable. If a test genuinely
    #    needs HOME isolated, it should set it explicitly in its own
    #    fixture. Any code in the codebase reading ``~/.hermes/*`` via
    #    ``Path.home() / ".hermes"`` instead of ``get_hermes_home()``
    #    is a bug to fix at the callsite.
    fake_hermes_home = tmp_path / "hermes_test"
    fake_hermes_home.mkdir()
    (fake_hermes_home / "sessions").mkdir()
    (fake_hermes_home / "cron").mkdir()
    (fake_hermes_home / "memories").mkdir()
    (fake_hermes_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_hermes_home))

    # 4. Deterministic locale / timezone / hashseed. CI runs in UTC with
    #    C.UTF-8 locale; local dev often doesn't. Pin everything.
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    # 4b. Disable AWS IMDS lookups. Without this, any test that ends up
    #     calling has_aws_credentials() / resolve_aws_auth_env_var()
    #     (e.g. provider auto-detect, status command, cron run_job) burns
    #     ~2s waiting for the metadata service at 169.254.169.254 to time
    #     out. Tests don't run on EC2 — IMDS is always unreachable here.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_METADATA_SERVICE_TIMEOUT", "1")
    monkeypatch.setenv("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")
    # Tirith auto-installs from GitHub when enabled and missing. Unit tests
    # should never perform that implicit network/bootstrap path; Tirith-specific
    # tests opt back in by patching the security config directly.
    monkeypatch.setenv("TIRITH_ENABLED", "false")

    # 5. Reset plugin singleton so tests don't leak plugins from
    #    ~/.hermes/plugins/ (which, per step 3, is now empty — but the
    #    singleton might still be cached from a previous test).
    try:
        import hermes_cli.plugins as _plugins_mod
        monkeypatch.setattr(_plugins_mod, "_plugin_manager", None)
    except Exception:
        pass
    # Explicitly clear provider-specific base URL overrides that don't match
    # the generic credential-shaped env-var filter above.
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.delenv("GMI_BASE_URL", raising=False)


# Backward-compat alias — old tests reference this fixture name. Keep it
# as a no-op wrapper so imports don't break.
@pytest.fixture(autouse=True)
def _isolate_hermes_home(_hermetic_environment):
    """Alias preserved for any test that yields this name explicitly."""
    return None


# ── Module-level state reset — replaced by per-file process isolation ──────
#
# Each test FILE runs in a freshly-spawned ``python -m pytest <file>``
# subprocess via ``scripts/run_tests_parallel.py``, so module-level dicts /
# sets / ContextVars from tests in one file cannot leak into tests in
# another file. No manual per-module clearing needed.
#
# Within a single file, ordering is the author's responsibility. If your
# tests in the same file share mutable state, either reset it explicitly
# in a fixture or split them across files.
#
# The skill ``test-suite-cascade-diagnosis`` documents the cascade patterns
# this replaces; the running example was ``test_command_guards`` failing
# 12/15 CI runs because ``tools.approval._session_approved`` carried
# approvals from one test's session into another's.


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up automatically."""
    return tmp_path


@pytest.fixture()
def mock_config():
    """Return a minimal hermes config dict suitable for unit tests."""
    return {
        "model": "test/mock-model",
        "toolsets": ["terminal", "file"],
        "max_turns": 10,
        "terminal": {
            "backend": "local",
            "cwd": "/tmp",
            "timeout": 30,
        },
        "compression": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "command_allowlist": [],
    }


# ── Per-test timeout — handled by the isolation plugin ─────────────────────
#
# The subprocess-per-test plugin enforces the configured ``isolate_timeout``
# ini key by terminating the child if it overruns. The old SIGALRM-based
# fixture (POSIX-only, didn't work on Windows) is gone.


@pytest.fixture(autouse=True)
def _ensure_current_event_loop(request):
    """Provide a default event loop for sync tests that call get_event_loop().

    Python 3.11+ no longer guarantees a current loop for plain synchronous tests.
    A number of gateway tests still use asyncio.get_event_loop().run_until_complete(...).
    Ensure they always have a usable loop without interfering with pytest-asyncio's
    own loop management for @pytest.mark.asyncio tests.

    On Python 3.12+, ``asyncio.get_event_loop_policy().get_event_loop()`` with no
    *running* loop emits DeprecationWarning; skip that path and install a fresh
    loop via ``new_event_loop()`` instead.
    """
    if request.node.get_closest_marker("asyncio") is not None:
        yield
        return

    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop is None and sys.version_info < (3, 12):
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None

    created = loop is None or loop.is_closed()
    if created:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield
    finally:
        if created and loop is not None:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)


# ── Live-system guard ──────────────────────────────────────────────────────
#
# Several test files exercise the gateway-restart / kill code paths
# (``cmd_update``, ``kill_gateway_processes``, ``stop_profile_gateway``).
# When a single test forgets to mock either ``os.kill`` or the global
# ``find_gateway_pids`` helper, the real call leaks out of the hermetic
# environment and finds the developer's live ``hermes-gateway`` process
# via ``psutil`` — sending it SIGTERM mid-test. The shutdown forensics in
# PR #23285 caught this happening 5+ times in 3 days, every time
# correlated with a ``tests/hermes_cli/`` pytest run starting up.
#
# This fixture makes the leak impossible by intercepting the two
# primitives that actually do damage:
#
#  • ``os.kill`` rejects any PID outside the test process subtree with
#    a hard ``RuntimeError`` so the offending test gets a stack trace
#    instead of silently murdering the real gateway.
#  • ``subprocess.run`` / ``subprocess.Popen`` / ``call`` / ``check_call`` /
#    ``check_output`` reject mutating systemd/launchd commands, detached
#    Hermes gateway spawns, and process-killer commands targeting a foreign
#    PID. Read-only service-manager calls still pass through outside updater
#    tests.
#
# We intentionally do NOT stub ``find_gateway_pids`` / ``_scan_gateway_pids``
# here — tests of those functions themselves need the real implementation.
# Even if a test gets the live gateway PID back from a real scan, the
# ``os.kill`` guard above catches the actual signal call, and the
# ``systemctl`` guard catches the systemd path. Discovery without
# delivery is harmless.

_LIVE_SYSTEM_GUARD_BYPASS_MARK = "live_system_guard_bypass"


def _guard_cmd_to_string(cmd) -> str:
    """Render an argv/string for pure, side-effect-free guard inspection."""
    if cmd is None:
        return ""
    if isinstance(cmd, (bytes, bytearray)):
        try:
            return bytes(cmd).decode(errors="replace")
        except Exception:
            return ""
    if isinstance(cmd, str):
        return cmd
    if isinstance(cmd, (list, tuple)):
        try:
            return " ".join(_guard_token_to_string(token) for token in cmd)
        except Exception:
            return ""
    return str(cmd)


def _guard_token_to_string(token) -> str:
    """Normalize one argv element without collapsing bytes into ``b'...'``."""
    if isinstance(token, (bytes, bytearray)):
        return bytes(token).decode(errors="replace")
    return str(token)


def _guard_tokens(cmd) -> list[str]:
    import shlex

    if isinstance(cmd, (list, tuple)):
        try:
            return [_guard_token_to_string(token) for token in cmd]
        except Exception:
            return []
    rendered = _guard_cmd_to_string(cmd)
    try:
        return shlex.split(rendered)
    except ValueError:
        return rendered.split()


def _guard_command_variants(cmd) -> list[list[str]]:
    """Return argv plus recursively parsed shell-wrapper payloads."""
    import base64

    posix_shells = {"sh", "bash", "dash", "zsh", "ash", "ksh"}
    windows_shells = {"cmd", "powershell", "pwsh"}
    protected_hints = ("hermes", "gateway")
    variants: list[list[str]] = []
    pending = [(_guard_tokens(cmd), 0)]
    while pending:
        tokens, depth = pending.pop()
        if not tokens:
            continue
        variants.append(tokens)
        if depth >= 5:
            continue
        for index, token in enumerate(tokens):
            head = _guard_executable(token)
            args = tokens[index + 1:]
            payload = None
            if head in posix_shells:
                for position, flag in enumerate(args[:-1]):
                    if (
                        flag.startswith("-")
                        and not flag.startswith("--")
                        and "c" in flag[1:].lower()
                    ):
                        payload = args[position + 1]
                        break
            elif head == "cmd":
                for position, flag in enumerate(args[:-1]):
                    if flag.lower() in {"/c", "/k"}:
                        payload = args[position + 1]
                        break
            elif head in windows_shells:
                for position, flag in enumerate(args[:-1]):
                    normalized_flag = flag.lower()
                    if normalized_flag in {"-command", "/command", "-c"}:
                        payload = args[position + 1]
                        break
                    if normalized_flag in {"-encodedcommand", "-e", "-enc"}:
                        encoded_payload = args[position + 1]
                        try:
                            payload = base64.b64decode(
                                encoded_payload, validate=True
                            ).decode("utf-16-le")
                        except (UnicodeDecodeError, ValueError):
                            if any(hint in encoded_payload.lower() for hint in protected_hints):
                                pending.append(
                                    (["__guard_decode_failure__", encoded_payload], depth + 1)
                                )
                        break
            if payload is not None:
                pending.append((_guard_tokens(payload), depth + 1))
    return variants


def _guard_executable(token: str) -> str:
    executable = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    return executable[:-4] if executable.endswith(".exe") else executable


def _is_service_manager_command(cmd) -> bool:
    """True for a systemd or launchd command, irrespective of its verb."""
    return any(
        _guard_executable(token) in {"systemctl", "launchctl"}
        for tokens in _guard_command_variants(cmd)
        for token in tokens
    )


def _is_service_manager_mutation_command(cmd) -> bool:
    """Fail closed for non-read-only verbs targeting a Hermes service."""
    hermes_tokens = (
        "hermes-gateway",
        "hermes.service",
        "ai.hermes.gateway",
        "hermes_cli.main gateway",
        "hermes_cli/main.py gateway",
        "gateway/run.py",
        "hermes gateway",
    )
    read_only_verbs = {
        "status", "show", "cat", "list-units", "list", "is-active",
        "is-enabled", "is-failed", "is-system-running", "print", "blame",
        "get-default",
    }
    option_values = {
        "--property", "--type", "--state", "--output", "--lines",
        "--job-mode", "--signal", "--kill-whom", "--what", "--host",
        "--machine", "--root", "--image", "--firmware-setup", "--timestamp",
    }
    short_options_with_values = {"-p", "-t", "-o", "-n", "-h", "-m"}
    known_flag_options = {
        "--all", "--ask-password", "--force", "--full", "--global", "--help",
        "--legend", "--no-ask-password", "--no-block", "--no-legend", "--no-pager",
        "--no-reload", "--now", "--plain", "--quiet", "--recursive", "--reverse",
        "--runtime", "--system", "--user", "--version", "-a", "-f", "-l", "-q",
        "-r",
    }

    def _positional_verb(tokens, manager_index):
        position = manager_index + 1
        while position < len(tokens):
            token = tokens[position]
            lowered = token.lower()
            if token == "--":
                return tokens[position + 1].lower() if position + 1 < len(tokens) else None
            if not token.startswith("-"):
                return lowered
            if "=" in token:
                option = lowered.split("=", 1)[0]
                if option in option_values or option in known_flag_options:
                    position += 1
                    continue
                return None
            if lowered in option_values or lowered in short_options_with_values:
                if position + 1 >= len(tokens):
                    return None
                position += 2
                continue
            if lowered in known_flag_options:
                position += 1
                continue
            return None
        return None

    for tokens in _guard_command_variants(cmd):
        low = " ".join(tokens).lower()
        if tokens and tokens[0] == "__guard_decode_failure__":
            if any(marker in low for marker in hermes_tokens):
                return True
            continue
        if not any(marker in low for marker in hermes_tokens):
            continue
        if _guard_executable(tokens[0]) in {
            "stop-service", "start-service", "restart-service", "set-service",
            "remove-service",
        }:
            return True
        for index, token in enumerate(tokens):
            if _guard_executable(token) not in {"systemctl", "launchctl"}:
                continue
            verb = _positional_verb(tokens, index)
            if verb not in read_only_verbs:
                return True
    return False


def _is_detached_gateway_spawn(name: str, cmd) -> bool:
    """Classify detached gateway launches without executing the command."""
    if name != "Popen":
        return False
    for tokens in _guard_command_variants(cmd):
        lower_tokens = [token.lower() for token in tokens]
        if "gateway" not in lower_tokens or "run" not in lower_tokens:
            continue
        low = " ".join(tokens).lower()
        if any(
            marker in low
            for marker in ("hermes_cli.main", "hermes_cli/main.py", "gateway/run.py")
        ):
            return True
        if any(_guard_executable(token) == "hermes" for token in tokens):
            return True
    return False


def _is_process_killer_command(cmd, *, is_own_subtree) -> bool:
    """Classify process-killer argv, including Windows PID/image kills."""
    process_killers = {"pkill", "killall", "taskkill", "skill", "fuser", "kill", "killpg"}
    for tokens in _guard_command_variants(cmd):
        if not tokens:
            continue
        heads = [_guard_executable(token) for token in tokens]
        low = " ".join(tokens).lower()
        for index, head in enumerate(heads):
            if head not in process_killers:
                continue
            if head == "taskkill":
                upper_tokens = [part.upper() for part in tokens]
                for flag in ("/IM", "-IM"):
                    if flag not in upper_tokens:
                        continue
                    image_index = upper_tokens.index(flag) + 1
                    image = tokens[image_index].lower() if image_index < len(tokens) else ""
                    if "hermes-gateway" in image or image.startswith("python"):
                        return True
                try:
                    pid_index = next(
                        position + 1
                        for position, part in enumerate(upper_tokens)
                        if part in {"/PID", "-PID"}
                    )
                    target_pid = int(tokens[pid_index])
                except (StopIteration, ValueError, IndexError):
                    target_pid = None
                if target_pid is not None and not is_own_subtree(target_pid):
                    return True
                continue
            if head in {"kill", "killpg"}:
                args = tokens[index + 1:]
                if head == "killpg":
                    if args and args[0] == "--":
                        args = args[1:]
                    if not args:
                        return True
                    try:
                        target_pid = int(args[0])
                    except ValueError:
                        return True
                    if target_pid <= 0 or not is_own_subtree(target_pid):
                        return True
                    continue

                position = 0
                if args and args[0] == "--":
                    position = 1
                elif args and args[0].startswith("-"):
                    signal_option = args[0]
                    if signal_option.lower() in {"-s", "-n"}:
                        position = 2
                    elif signal_option[1:].isalnum():
                        position = 1
                    else:
                        return True
                if position < len(args) and args[position] == "--":
                    position += 1
                targets = args[position:]
                if not targets:
                    return True
                for target in targets:
                    try:
                        target_pid = int(target)
                    except ValueError:
                        return True
                    if target_pid <= 0 or not is_own_subtree(target_pid):
                        return True
                continue
            if (
                "hermes" in low
                or "gateway" in low
                or ("python" in low and "-f" in tokens)
            ):
                return True
    return False


def pytest_configure(config):  # noqa: D401 — pytest hook
    """Register markers used by hermetic conftest."""
    config.addinivalue_line(
        "markers",
        f"{_LIVE_SYSTEM_GUARD_BYPASS_MARK}: bypass the live-system guard "
        "(only for tests that genuinely need real os.kill / subprocess "
        "behaviour — e.g. PTY tests that signal their own child).",
    )

    # The pyproject addopts pin ``--timeout-method=signal`` relies on
    # ``signal.SIGALRM``, which does not exist on Windows — pytest-timeout
    # raises AttributeError at timer setup and the whole run aborts before any
    # test executes. Fall back to the thread-based timer on Windows so the
    # suite runs natively there (POSIX keeps the more reliable signal method).
    if sys.platform == "win32" and getattr(config.option, "timeout_method", None) == "signal":
        config.option.timeout_method = "thread"


@pytest.fixture(autouse=True)
def _live_system_guard(request, monkeypatch, _hermetic_environment):
    """Block real os.kill / systemctl / gateway-pid scans during tests.

    See block comment above for the why. Tests that genuinely need
    real signal delivery (e.g. PTY tests that SIGINT their own child)
    can opt out with ``@pytest.mark.live_system_guard_bypass``.

    Coverage (every primitive that can deliver a signal to or otherwise
    terminate a foreign process):
      • os.kill, os.killpg (POSIX)
      • subprocess.run / Popen / call / check_call / check_output
      • subprocess.getoutput / getstatusoutput
      • os.system / os.popen
      • pty.spawn
      • asyncio.create_subprocess_exec / create_subprocess_shell
    Subprocess inspection looks at the WHOLE command string (not just
    tokens[0]), so ``bash -c "systemctl restart hermes-gateway"``,
    ``sudo systemctl ...``, ``env systemctl ...``, ``setsid systemctl ...``
    are all caught. ``pkill``/``killall``/``taskkill`` invocations
    targeting hermes/python patterns are also blocked.
    """
    if request.node.get_closest_marker(_LIVE_SYSTEM_GUARD_BYPASS_MARK):
        yield
        return

    import os as _os
    import subprocess as _subprocess

    # ``cmd_update`` discovers and restarts gateways after otherwise-mocked
    # git/dependency work.  Keep discovery tests real, but return inert values
    # for the entire updater stack so a missing mock cannot find a developer's
    # running gateway or schedule a detached restart watcher.
    from pathlib import Path as _Path

    def _inside_cmd_update() -> bool:
        frame = sys._getframe(1)
        while frame is not None:
            if (
                frame.f_code.co_name in {"cmd_update", "_cmd_update_impl"}
                and frame.f_globals.get("__name__") == "hermes_cli.main"
            ):
                return True
            frame = frame.f_back
        return False

    def _inert_during_update(real, inert):
        def _guarded(*args, **kwargs):
            if _inside_cmd_update():
                return inert()
            return _guarded.__wrapped__(*args, **kwargs)

        _guarded.__wrapped__ = real
        _guarded._live_system_guard_inert = True
        _guarded.__name__ = getattr(real, "__name__", "_guarded")
        return _guarded

    # Do not import ``hermes_cli.main`` here: importing it resolves profiles,
    # dotenv/config, and logging for every test-file subprocess. An updater
    # test has it loaded at collection time, so only then load the mutation
    # boundaries that need updater-scoped inerting.
    if "hermes_cli.main" in sys.modules:
        from hermes_cli import gateway as _gateway
        from hermes_cli import gateway_windows as _gateway_windows
        from gateway import status as _gateway_status

        def _inert_unit_refresh(path_getter):
            def _guarded_refresh(*args, **kwargs):
                target = path_getter(*args, **kwargs).resolve()
                hermes_home = _Path(os.environ["HERMES_HOME"]).resolve()
                if target != hermes_home and hermes_home not in target.parents:
                    raise RuntimeError(
                        "tests/conftest.py live-system guard: blocked updater "
                        f"unit write to non-test path {target}"
                    )
                return False

            return _guarded_refresh

        for _name, _inert in (
            ("find_gateway_pids", list),
            ("_scan_gateway_pids", list),
            ("_get_service_pids", set),
            ("find_profile_gateway_processes", list),
            ("supports_systemd_services", lambda: False),
            ("is_macos", lambda: False),
            ("_ensure_user_systemd_env", lambda: None),
            ("launchd_restart", lambda: None),
            ("get_launchd_label", lambda: "ai.hermes.gateway"),
            (
                "get_launchd_plist_path",
                lambda: _Path("/definitely-not-a-real-hermes-launchd.plist"),
            ),
            ("launch_detached_profile_gateway_restart", lambda: False),
            ("launch_detached_gateway_restart_by_cmdline", lambda: False),
            ("_capture_gateway_argv", lambda: None),
            (
                "refresh_systemd_unit_if_needed",
                _inert_unit_refresh(
                    lambda *args, **kwargs: _gateway.get_systemd_unit_path(
                        system=kwargs.get("system", args[0] if args else False)
                    )
                ),
            ),
            (
                "refresh_launchd_plist_if_needed",
                _inert_unit_refresh(lambda *args, **kwargs: _gateway.get_launchd_plist_path()),
            ),
        ):
            monkeypatch.setattr(
                _gateway,
                _name,
                _inert_during_update(getattr(_gateway, _name), _inert),
            )
        for _name, _inert in (
            ("is_installed", lambda: False),
            ("_spawn_detached", lambda: None),
        ):
            monkeypatch.setattr(
                _gateway_windows,
                _name,
                _inert_during_update(getattr(_gateway_windows, _name), _inert),
            )
        monkeypatch.setattr(
            _gateway_status,
            "terminate_pid",
            _inert_during_update(_gateway_status.terminate_pid, lambda: None),
        )

    # Follow-up: psutil.Process.terminate/kill/send_signal are intentionally
    # out of scope; updater termination is inerted above via terminate_pid.

    test_pid = _os.getpid()
    # Capture the test process's existing children at fixture start —
    # any *new* children spawned by the test are also allowlisted via
    # the live psutil walk below. Static set keeps the fast path cheap.
    try:
        import psutil as _psutil
        _initial_children = {
            c.pid for c in _psutil.Process(test_pid).children(recursive=True)
        }
    except Exception:
        _psutil = None
        _initial_children = set()

    def _is_own_subtree(pid: int) -> bool:
        # PID 0 means "our own process group"; -1 means "every process we
        # can signal". Both are dangerous when paired with SIGTERM/SIGKILL,
        # but pid 0 is technically scoped to our group so allow it; pid -1
        # is treated as foreign (refuse).
        if pid == 0:
            return True
        if pid < 0:
            return False
        # PID 1 is the self-test's designated foreign PID.  A namespaced
        # pytest runner can itself be PID 1, but treating init as an owned
        # child would turn the guard's foreign-PID proof into a real killpg.
        if pid == 1:
            return False
        if pid == test_pid or pid in _initial_children:
            return True
        if _psutil is None:
            return False
        try:
            walker = _psutil.Process(pid)
        except Exception:
            # Stale PID — kill would be a no-op anyway, allow it.
            return True
        try:
            for parent in walker.parents():
                if parent.pid == test_pid:
                    return True
        except Exception:
            return False
        return False

    real_kill = _os.kill

    def _guarded_kill(pid, sig, *args, **kwargs):
        # Signal 0 is a pure liveness probe — it cannot terminate anything.
        # psutil.pid_exists() uses os.kill(pid, 0) on POSIX, and probing a
        # just-killed grandchild that was reparented to init (zombie with a
        # foreign parent chain) must not trip the guard. Flaked in CI on
        # test_entire_tree_is_sigkilled_not_just_parent.
        if int(sig) == 0:
            return real_kill(pid, sig, *args, **kwargs)
        if _is_own_subtree(int(pid)):
            return real_kill(pid, sig, *args, **kwargs)
        raise RuntimeError(
            f"tests/conftest.py live-system guard: blocked os.kill("
            f"{pid}, {sig}) — PID is outside the test process subtree. "
            "If this fired in CI it means the test reached a real "
            "kill_gateway_processes / stop_profile_gateway / cmd_update "
            "code path without mocking find_gateway_pids and os.kill. "
            "Mock both, or mark the test with "
            "@pytest.mark.live_system_guard_bypass if real signal "
            "delivery is genuinely required."
        )

    monkeypatch.setattr(_os, "kill", _guarded_kill)

    # ``os.killpg`` is the same risk class — sends a signal to every
    # process in a group. The gateway is a session leader (its own
    # PGID == its PID), so killpg(gateway_pid, SIGTERM) is a one-shot
    # kill of the live process. Allow it only when the target PGID is
    # the test process's own group.
    if hasattr(_os, "killpg"):
        real_killpg = _os.killpg
        own_pgid = _os.getpgrp()

        def _guarded_killpg(pgid, sig, *args, **kwargs):
            # Signal 0 is a pure liveness probe — never destructive.
            if int(sig) == 0:
                return real_killpg(pgid, sig, *args, **kwargs)
            # In a PID namespace pytest can share init's group.  PID 1 is
            # nevertheless the foreign-PID safety sentinel and must never be
            # considered an owned test group for destructive signals.
            if int(pgid) == 1:
                raise RuntimeError(
                    f"tests/conftest.py live-system guard: blocked "
                    f"os.killpg({pgid}, {sig}) — PID 1 is always foreign."
                )
            if int(pgid) == own_pgid or _is_own_subtree(int(pgid)):
                return real_killpg(pgid, sig, *args, **kwargs)
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"os.killpg({pgid}, {sig}) — PGID is outside the test "
                "process group. See _live_system_guard for the why."
            )

        monkeypatch.setattr(_os, "killpg", _guarded_killpg)

    # ── Subprocess command-string inspection (whole-line) ──────────
    def _check_subprocess_cmd(name, cmd):
        if _is_service_manager_mutation_command(cmd):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — would mutate the "
                "live hermes-gateway service definition. Mock "
                "subprocess.run / _run_systemctl in the test, or "
                "mark with @pytest.mark.live_system_guard_bypass."
            )
        if _is_detached_gateway_spawn(name, cmd):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — would launch a Hermes "
                "gateway outside the test process lifecycle. Mock Popen "
                "in the test, or mark with "
                "@pytest.mark.live_system_guard_bypass."
            )
        if _is_process_killer_command(cmd, is_own_subtree=_is_own_subtree):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — process-killer command "
                "targeting hermes/python could hit the live gateway. "
                "Mark with @pytest.mark.live_system_guard_bypass if "
                "intentional."
            )
        # Block any subprocess that would run `hermes update` (or the
        # equivalent `python -m hermes_cli.main update`).  These commands
        # run `git fetch origin + git pull` against the REAL checkout,
        # overwriting files like pyproject.toml mid-test-run and corrupting
        # every subsequent subprocess that reads them.  The corruption is
        # especially insidious because the spawned process uses setsid/
        # start_new_session=True, making it invisible to pytest's process
        # tree (PPid=1) and nearly impossible to trace without explicit
        # inotify/SHA watchdogs.  Any test that legitimately needs to exercise
        # the update-spawn path must mock subprocess.Popen explicitly.
        cmd_str = _guard_cmd_to_string(cmd)
        low = cmd_str.lower()
        if "update" in low and (
            # hermes update / hermes update --gateway / setsid bash -c ... hermes update
            ("hermes" in low and "update" in low.split())
            or
            # python -m hermes_cli.main update --gateway
            ("hermes_cli" in low and "update" in low.split())
            or
            # venv/bin/hermes update  (absolute path variant used in tests)
            (".venv/bin/hermes" in low and "update" in low)
        ):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — this command would run "
                "`hermes update` against the real checkout, fetching "
                "from origin and overwriting repo files (e.g. "
                "pyproject.toml) mid-test-run. This corrupts every "
                "subsequent subprocess in the same runner. "
                "Mock subprocess.Popen (and subprocess.run if used) "
                "in the test instead, or mark with "
                "@pytest.mark.live_system_guard_bypass if genuinely "
                "needed (e.g. an integration test testing the update "
                "flow against a dedicated throwaway repo)."
            )

    def _wrap_subprocess(name, real):
        def _guarded(cmd, *args, **kwargs):
            if name == "run" and _inside_cmd_update() and _is_service_manager_command(cmd):
                return _subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
            _check_subprocess_cmd(name, cmd)
            return _guarded.__wrapped__(cmd, *args, **kwargs)
        _guarded.__name__ = f"_guarded_{name}"
        _guarded.__wrapped__ = real
        # Make the wrapper subscriptable like the wrapped callable when
        # the wrapped object is. ``subprocess.Popen[bytes]`` is used as
        # a type annotation in third-party packages (mcp, etc.); replacing
        # ``Popen`` with a plain function breaks ``Popen[bytes]`` at
        # import time. Defer ``__class_getitem__`` to the original.
        if hasattr(real, "__class_getitem__"):
            _guarded.__class_getitem__ = real.__class_getitem__
        return _guarded

    def _wrap_popen():
        """Subclass Popen so isinstance checks AND Popen[bytes] still work."""
        real = _subprocess.Popen

        class _GuardedPopen(real):  # type: ignore[misc, valid-type]
            def __init__(self, cmd, *args, **kwargs):
                _check_subprocess_cmd("Popen", cmd)
                super().__init__(cmd, *args, **kwargs)

        _GuardedPopen.__name__ = "Popen"
        _GuardedPopen.__qualname__ = "Popen"
        return _GuardedPopen

    real_run = _subprocess.run
    real_popen = _subprocess.Popen
    real_call = _subprocess.call
    real_check_call = _subprocess.check_call
    real_check_output = _subprocess.check_output
    real_getoutput = _subprocess.getoutput
    real_getstatusoutput = _subprocess.getstatusoutput

    monkeypatch.setattr(_subprocess, "run", _wrap_subprocess("run", real_run))
    monkeypatch.setattr(_subprocess, "Popen", _wrap_popen())
    monkeypatch.setattr(_subprocess, "call", _wrap_subprocess("call", real_call))
    monkeypatch.setattr(
        _subprocess, "check_call", _wrap_subprocess("check_call", real_check_call)
    )
    monkeypatch.setattr(
        _subprocess,
        "check_output",
        _wrap_subprocess("check_output", real_check_output),
    )
    monkeypatch.setattr(
        _subprocess, "getoutput", _wrap_subprocess("getoutput", real_getoutput)
    )
    monkeypatch.setattr(
        _subprocess,
        "getstatusoutput",
        _wrap_subprocess("getstatusoutput", real_getstatusoutput),
    )

    # os.system / os.popen — same risk class, completely unwrapped before.
    real_os_system = _os.system
    real_os_popen = _os.popen

    def _guarded_os_system(command):
        _check_subprocess_cmd("os.system", command)
        return real_os_system(command)

    def _guarded_os_popen(cmd, *args, **kwargs):
        _check_subprocess_cmd("os.popen", cmd)
        return real_os_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(_os, "system", _guarded_os_system)
    monkeypatch.setattr(_os, "popen", _guarded_os_popen)

    # pty.spawn — POSIX-only.
    try:
        import pty as _pty
        if hasattr(_pty, "spawn"):
            real_pty_spawn = _pty.spawn

            def _guarded_pty_spawn(argv, *args, **kwargs):
                _check_subprocess_cmd("pty.spawn", argv)
                return real_pty_spawn(argv, *args, **kwargs)

            monkeypatch.setattr(_pty, "spawn", _guarded_pty_spawn)
    except Exception:
        pass

    # asyncio.create_subprocess_* — bypasses subprocess module entirely.
    try:
        import asyncio as _asyncio
        real_async_exec = _asyncio.create_subprocess_exec
        real_async_shell = _asyncio.create_subprocess_shell

        async def _guarded_async_exec(program, *args, **kwargs):
            _check_subprocess_cmd(
                "asyncio.create_subprocess_exec", [program, *args]
            )
            return await real_async_exec(program, *args, **kwargs)

        async def _guarded_async_shell(cmd, *args, **kwargs):
            _check_subprocess_cmd("asyncio.create_subprocess_shell", cmd)
            return await real_async_shell(cmd, *args, **kwargs)

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _guarded_async_exec)
        monkeypatch.setattr(
            _asyncio, "create_subprocess_shell", _guarded_async_shell
        )
    except Exception:
        pass

    yield

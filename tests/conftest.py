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
import atexit
import importlib
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Sandbox HERMES_HOME before ANY test module is imported ──────────────────
# `hermes_cli/main.py` calls `setup_logging()` at MODULE level, which resolves
# `get_hermes_home()` and attaches rotating file handlers to the ROOT logger.
# So merely importing it - which many test modules do, directly or
# transitively - points the whole pytest session's logging at the operator's
# real `~/.hermes/logs/agent.log` and `errors.log`.
#
# The `_isolate_env` fixture below also sandboxes HERMES_HOME, but fixtures run
# AFTER collection imports test modules, by which point the handler already
# holds an absolute path to the real log. Measured on a live install: 126
# warnings in the operator's agent.log came from test runs, not the gateway -
# enough noise to make genuine warnings hard to find.
#
# conftest is imported before any test module, so setting it here closes that
# window. The per-test fixture still applies for everything after import.
#
# ORDER MATTERS: the kanban write guard's deny-list (further down) must know
# the REAL Hermes root — capture it BEFORE the sandbox rewires HERMES_HOME,
# otherwise the deny-list would point at the throwaway tempdir and the guard
# would silently stop protecting the operator's actual ~/.hermes (#69385).
_PRE_SANDBOX_KANBAN_OVERRIDE = os.environ.get("HERMES_KANBAN_HOME", "").strip()
_PRE_SANDBOX_HERMES_HOME = os.environ.get("HERMES_HOME", "")


def _hermes_home_points_at_production(value: str) -> bool:
    """True when a pre-set HERMES_HOME resolves to the real production root.

    Gateway-launched shells (and developer shells that ``export
    HERMES_HOME=~/.hermes``) hand pytest the PRODUCTION home. Historically
    the session sandbox below honored any pre-set value, so collection-time
    imports (logging handlers, ``hermes_state.DEFAULT_DB_PATH``) froze paths
    inside the real ``~/.hermes`` — the escape vector that landed pytest
    fixture rows (chat-1 / wx-chat sessions, /tmp/pytest-of-* routing
    scopes) in the live state.db and flipped its journal mode under the
    WAL-mode gateway writer. Only a genuinely custom (non-production)
    HERMES_HOME is honored now.
    """
    if not value:
        return True
    try:
        resolved = Path(value).expanduser().resolve()
        real_root = (Path.home() / ".hermes").resolve()
    except Exception:
        return True
    if resolved == real_root:
        return True
    # Profile home directly under the production root: <root>/profiles/<name>
    return resolved.parent.name == "profiles" and resolved.parent.parent == real_root


if _hermes_home_points_at_production(os.environ.get("HERMES_HOME", "")):
    _SESSION_HERMES_HOME = tempfile.mkdtemp(prefix="hermes-test-home-")
    os.environ["HERMES_HOME"] = _SESSION_HERMES_HOME
    atexit.register(shutil.rmtree, _SESSION_HERMES_HOME, True)

# Subprocess-surviving isolation marker (#82770). PYTEST_CURRENT_TEST /
# PYTEST_VERSION are pytest's own vars, and tests that spawn children
# routinely rebuild the child env and strip them ("the subprocess must look
# like a real CLI") — which used to disarm hermes_state's live-DB guard in
# the child at the same moment the child lost the HERMES_HOME redirect.
# HERMES_TEST_ISOLATION is OUR marker: exported here (before any test module
# imports), inherited by every child by default, and honored by
# hermes_state_guard._running_under_pytest() as a test-context signal. A child
# that carries it and still resolves the production state.db fails hard.
# Tests that legitimately need a child to look like a non-test process AND
# open a real DB must export HERMES_STATE_DB_GUARD_BYPASS=1 in that child's
# env instead of stripping markers.
os.environ["HERMES_TEST_ISOLATION"] = os.environ.get("HERMES_HOME", "") or "1"

#: HERMES_HOME as it stood when conftest was imported - i.e. before any test
#: module could import code that configures logging. Recorded so the guard in
#: tests/test_log_isolation.py can assert the sandbox existed AT THAT MOMENT.
#: Reading os.environ from inside a test is useless here: the per-test
#: `_isolate_env` fixture has sandboxed it by then, so the check would pass
#: even with this block removed.
HERMES_HOME_AT_CONFTEST_IMPORT = os.environ.get("HERMES_HOME", "")


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
    "PERPLEXITY_API_KEY",
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
    "AI_GATEWAY_API_KEY",
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
    "AI_GATEWAY_BASE_URL",
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
    # Voice/TTS runtime flags. ``tui_gateway/server.py`` reads these straight
    # off ``os.environ`` at call time (``_voice_mode_enabled`` /
    # ``_voice_tts_enabled``) and, on every completed turn, hands the turn's
    # final response text to ``hermes_cli.voice.speak_text`` — real synthesis,
    # real playback, out of the developer's speakers. Blank them per-test so a
    # leak (from the shell, or from an earlier test that drove the
    # ``voice.toggle`` RPC, which writes ``os.environ`` directly) cannot carry
    # into the next test. See ``_audio_playback_guard`` for the second layer.
    "HERMES_VOICE",
    "HERMES_VOICE_TTS",
    "HERMES_YOLO_MODE",
    # Injected into subprocess envs by the terminal tool (_make_run_env), so
    # any test run launched FROM a Hermes agent session inherits them and
    # hermes_constants home-resolution helpers prefer them over monkeypatched
    # HOME (test_subprocess_home_isolation red locally, green on CI).
    "HERMES_REAL_HOME",
    "TERMINAL_HOME_MODE",
    "HERMES_INTERACTIVE",
    "HERMES_QUIET",
    "HERMES_TOOL_PROGRESS",
    "HERMES_TOOL_PROGRESS_MODE",
    "HERMES_MAX_ITERATIONS",
    "HERMES_SESSION_PLATFORM",
    "HERMES_SESSION_CHAT_ID",
    "HERMES_SESSION_CHAT_NAME",
    "HERMES_SESSION_CHAT_TYPE",
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
    # Pytest is routinely launched from a delegated worker.  The worker
    # lineage marker must not make parent-state tests run as delegated
    # children; tests that exercise child behavior set it explicitly.
    "HERMES_DELEGATED_CHILD_CONTEXT",
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
    "TERMINAL_VERCEL_RUNTIME",
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
    "TELEGRAM_GROUP_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_CHATS",
    "QQ_ALLOWED_USERS",
    "QQ_GROUP_ALLOWED_USERS",
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
    "PHOTON_ALLOWED_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    "TELEGRAM_ALLOW_ALL_USERS",
    "DISCORD_ALLOW_ALL_USERS",
    "WHATSAPP_ALLOW_ALL_USERS",
    "SLACK_ALLOW_ALL_USERS",
    "SIGNAL_ALLOW_ALL_USERS",
    "EMAIL_ALLOW_ALL_USERS",
    "SMS_ALLOW_ALL_USERS",
    "PHOTON_ALLOW_ALL_USERS",
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
    "PHOTON_HOME_CHANNEL",
    "PHOTON_HOME_CHANNEL_THREAD_ID",
    "PHOTON_HOME_CHANNEL_NAME",
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
    "SLACK_THREAD_REQUIRE_MENTION",
    "SLACK_IGNORE_OTHER_USER_MENTIONS",
    "SLACK_REQUIRE_MENTION_CHANNELS",
    "SLACK_FREE_RESPONSE_CHANNELS",
    "SLACK_ALLOWED_CHANNELS",
    "SLACK_IGNORED_CHANNELS",
    "SLACK_DISABLE_DMS",
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
    # Keep the subprocess-surviving isolation marker pointed at THIS test's
    # home (#82770): children spawned by the test inherit it by default, so
    # hermes_state's live-DB guard stays armed in them even when the test
    # strips pytest's own PYTEST_* vars from the child env.
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(fake_hermes_home))
    # And never let a developer-shell (or leaked child) bypass disarm the
    # guard for in-process code under test.
    monkeypatch.delenv("HERMES_STATE_DB_GUARD_BYPASS", raising=False)

    # 3b. hermes_state computes ``DEFAULT_DB_PATH = get_hermes_home() / "state.db"``
    #     at import time. When the module is first imported at collection (any
    #     test file with a top-level ``from hermes_state import ...``) that
    #     happens BEFORE this fixture ever runs, so every argless
    #     ``SessionDB()`` in every test opens the developer's REAL state.db —
    #     reading real sessions into assertions and writing test rows into the
    #     real profile. Re-pin the constant to this test's home. (Several test
    #     files already do this locally; this makes it an invariant.)
    hermes_state_mod = sys.modules.get("hermes_state")
    if hermes_state_mod is not None and hasattr(hermes_state_mod, "DEFAULT_DB_PATH"):
        monkeypatch.setattr(
            hermes_state_mod, "DEFAULT_DB_PATH", fake_hermes_home / "state.db"
        )

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
    # Lazy feature deps (tools/lazy_deps.py) pip-install on demand by design —
    # _allow_lazy_installs() fails open for users. Unit tests must never reach
    # pip/the network: with the SDK absent, any agent init whose tool checks
    # touch a lazy feature (e.g. check_tts_requirements →
    # ensure("tts.elevenlabs")) spawns a real pip install — which hangs to the
    # suite timeout under tests that set fake proxy env vars. The kill-switch
    # makes ensure() raise FeatureUnavailable immediately instead.
    # tests/tools/test_lazy_deps.py overrides this var in both directions.
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

    # 5. Reset plugin singleton so tests don't leak plugins from
    #    ~/.hermes/plugins/ (which, per step 3, is now empty — but the
    #    singleton might still be cached from a previous test).
    try:
        import hermes_cli.plugins as _plugins_mod
        monkeypatch.setattr(_plugins_mod, "_plugin_manager", None)
        # Also clear the keyed per-home manager cache (and any plugin
        # submodules it left in sys.modules) so a manager built for a
        # previous test's tmp_path HERMES_HOME can't leak forward. Paths
        # are unique per test, so collisions are unlikely, but a full
        # reset keeps this fixture the single source of plugin-state
        # hygiene rather than relying on path uniqueness.
        _plugins_mod._reset_plugin_managers_for_tests()
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


@pytest.fixture(autouse=True)
def _neutralize_kanban_memory_guard(request, monkeypatch):
    """Pin the kanban dispatcher's memory guard to "no data" for every test.

    The dispatcher consults live system memory before spawning (OOF-30/
    OOF-77: memory-derived default cap + pressure-based spawn restriction).
    Left un-patched, dispatch tests would pass or fail based on how loaded
    the CI runner happens to be. Defaulting the sample to ``{}`` makes the
    derived cap ``None`` and the pressure level ``"unknown"`` — i.e. the
    pre-guard behaviour every existing test was written against. Tests that
    exercise the guard itself opt out with
    ``@pytest.mark.real_memory_guard`` or patch the seam directly.
    """
    if request.node.get_closest_marker("real_memory_guard"):
        return
    try:
        from hermes_cli import kanban_db_dispatch as _kbd_mod
    except Exception:
        return
    monkeypatch.setattr(_kbd_mod, "_system_memory_sample", lambda: {}, raising=False)


@pytest.fixture(autouse=True)
def _neutralize_webbrowser(monkeypatch):
    """Record browser-open attempts instead of opening real browser windows."""
    import webbrowser as _webbrowser

    opened: list[object] = []

    def _record(url=None, *_args, **_kwargs):
        opened.append(url)
        return True

    class _RecordingBrowser:
        def open(self, url, *_args, **_kwargs):
            return _record(url)

        def open_new(self, url, *_args, **_kwargs):
            return _record(url)

        def open_new_tab(self, url, *_args, **_kwargs):
            return _record(url)

    browser = _RecordingBrowser()

    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(_webbrowser, name, _record, raising=False)
    monkeypatch.setattr(_webbrowser, "get", lambda *_args, **_kwargs: browser)

    return opened


@pytest.fixture(autouse=True)
def _neutralize_macos_keychain_creds(request, monkeypatch):
    """Default Anthropic credential resolution away from the real macOS Keychain."""
    if request.node.get_closest_marker(_ALLOW_MACOS_KEYCHAIN_MARK):
        return None

    try:
        _mod = importlib.import_module("agent.anthropic_credentials")
    except Exception:
        return None
    monkeypatch.setattr(
        _mod,
        "_read_claude_code_credentials_from_keychain",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    return None


# ── Kanban write guard (#69283) ─────────────────────────────────────────────
# When hermetic isolation is bypassed (stale checkout, wrong rootdir, direct
# invocation), kanban writes silently pollute the real ~/.hermes. This autouse
# fixture patches ``kanban_db_connect.connect`` to refuse writes whose resolved DB
# path lands under the REAL kanban root (captured at import time, before any
# fixture rewires the environment). A deny-list is used instead of an
# allow-list because test-level fixtures legitimately move HERMES_HOME to
# sibling directories — an allow-list captured at setup time would see the
# stale autouse-set value and falsely reject hermetic tests (#69385 review).


def _capture_real_kanban_root() -> Path:
    """Resolve the REAL kanban root from the pre-test environment.

    Uses the pre-sandbox environment snapshot taken at the very top of this
    file (before the session HERMES_HOME sandbox rewired the env), so the
    deny-list keeps pointing at the operator's actual root. Mirrors
    ``kanban_db.kanban_home()`` resolution order:
    1. ``HERMES_KANBAN_HOME`` env var when set and non-empty
    2. the real (pre-sandbox) Hermes root otherwise
    """
    if _PRE_SANDBOX_KANBAN_OVERRIDE:
        return Path(_PRE_SANDBOX_KANBAN_OVERRIDE).expanduser().resolve()
    if _PRE_SANDBOX_HERMES_HOME and not _hermes_home_points_at_production(
        _PRE_SANDBOX_HERMES_HOME
    ):
        # HERMES_HOME was genuinely set to a CUSTOM root before the sandbox
        # (production-pointing values are sandboxed away above, in which case
        # the env still holds the tempdir and the resolver would be wrong) —
        # honor it via the normal resolver (it may be a profile dir whose
        # root matters).
        from hermes_constants import get_default_hermes_root
        return get_default_hermes_root().resolve()
    # No pre-existing HERMES_HOME: the real root is the platform default,
    # NOT the sandbox tempdir now sitting in the env.
    return (Path.home() / ".hermes").resolve()


_REAL_KANBAN_ROOT = _capture_real_kanban_root()


@pytest.fixture(autouse=True)
def _kanban_write_guard(_hermetic_environment, monkeypatch):
    """Fail-closed guard: refuse kanban writes that target the REAL root.

    Uses a **deny-list**: only blocks writes where the resolved DB path
    (explicit ``db_path`` or ``kanban_db_path()``) lands under the real
    ``~/.hermes`` captured at import time. Hermetic tests that legitimately
    move HERMES_HOME to sibling tempdirs are unaffected.

    Only patches when ``hermes_cli.kanban_db_connect`` is *already imported*
    — a ``sys.modules`` probe, not an import — so the guard never drags the
    kanban module into unrelated test processes.

    Uses ``monkeypatch.setattr`` so pytest restores ``connect`` automatically
    after each test (no stacked wrappers or state leakage across tests).
    """
    _kdb = sys.modules.get("hermes_cli.kanban_db")
    _kdbc = sys.modules.get("hermes_cli.kanban_db_connect")
    if _kdb is None or _kdbc is None:
        return

    # The sys.modules probe can observe the module MID-IMPORT: a fixture
    # boundary firing while another test's lazy `import hermes_cli.kanban_db`
    # is still executing sees a partially initialized module whose `connect`
    # doesn't exist yet (AttributeError flake, caught in a full-suite run).
    # A half-imported module has no callers yet either — nothing to guard
    # this round; the next test's fixture will patch the completed module.
    _orig_connect = getattr(_kdbc, "connect", None)
    if _orig_connect is None or getattr(_kdb, "kanban_db_path", None) is None:
        return

    def _guarded_connect(db_path=None, *args, **kwargs):
        if db_path is not None:
            resolved = Path(db_path).expanduser().resolve()
        else:
            resolved = (
                _kdb.kanban_db_path(board=kwargs.get("board"))
                .expanduser()
                .resolve()
            )
        try:
            resolved.relative_to(_REAL_KANBAN_ROOT)
        except ValueError:
            # Resolved path is NOT under the real root — safe to write.
            return _orig_connect(db_path, *args, **kwargs)
        raise RuntimeError(
            f"kanban_write_guard: kanban DB path resolved to {resolved}, "
            f"which is under the REAL kanban root ({_REAL_KANBAN_ROOT}). "
            f"Hermetic isolation has been bypassed — refusing to write "
            f"to the real ~/.hermes. See #69283."
        )

    monkeypatch.setattr(_kdbc, "connect", _guarded_connect)


# ── Live state.db write guard ───────────────────────────────────────────────
# Companion to the kanban guard above, for the MAIN state database.
# ``hermes_state._ensure_test_isolation`` (the single choke point every
# ``SessionDB()`` construction goes through) refuses, under pytest, any DB
# path that resolves inside the REAL Hermes root. This fixture wires the
# test-side knobs:
#   • honors ``@pytest.mark.live_system_guard_bypass`` (the established
#     escape-hatch marker) by disabling the state-db guard for that test;
#   • injects the pre-sandbox CUSTOM production root (Docker/portable
#     installs where HERMES_HOME is not ~/.hermes) into the guard's
#     deny-list, mirroring the kanban deny-list capture above.
# The guard itself is env-activated (PYTEST_CURRENT_TEST / PYTEST_VERSION),
# so subprocess children that import hermes_state directly are covered even
# without this fixture.


@pytest.fixture(autouse=True)
def _state_db_write_guard(request, monkeypatch):
    _hs = sys.modules.get("hermes_state")
    if _hs is None or not hasattr(_hs, "_STATE_DB_GUARD_BYPASS"):
        yield
        return
    if request.node.get_closest_marker("live_system_guard_bypass") is not None:
        monkeypatch.setattr(_hs, "_STATE_DB_GUARD_BYPASS", True)
        yield
        return
    extra_roots = []
    if _PRE_SANDBOX_HERMES_HOME and not _hermes_home_points_at_production(
        _PRE_SANDBOX_HERMES_HOME
    ):
        extra_roots.append(
            Path(_PRE_SANDBOX_HERMES_HOME).expanduser().resolve()
        )
    monkeypatch.setattr(
        _hs, "_STATE_DB_GUARD_EXTRA_DENY_ROOTS", tuple(extra_roots)
    )
    yield


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


# ── tui_gateway.server shared-module state isolation ───────────────────────
#
# ``tui_gateway.server`` registers its RPC handlers in a module-level
# ``_methods`` dict at import time and keeps per-session state in module
# globals (sessions, child-run registry, config cache, DB handle). The
# canonical per-file process isolation above hides any leakage, but a direct
# multi-file invocation (``pytest tests/tui_gateway/ tests/test_tui_gateway_server.py``,
# or plain ``pytest tests/``) shares one interpreter: a test that stubs
# ``_methods["slash.exec"]`` or leaves an active-session lease behind breaks
# unrelated tests in later files. This fixture snapshots the cheap-to-copy
# globals before each test and restores them after, so any file combination
# is order-independent. It is a near no-op (one sys.modules lookup) while
# the module has not been imported.
#
# The case this cannot cover — the module is first imported *during* a test
# that also mutates ``_methods`` — is handled by the importing files' own
# ``server`` fixtures (tests/tui_gateway/test_protocol.py and friends), which
# snapshot immediately after the import.

_TUI_SERVER_MODULE = "tui_gateway.server"


def _teardown_tui_server_sessions(mod) -> None:
    """Close leftover sessions through the production teardown boundary.

    Besides returning active-session leases, this finalizes the session,
    unregisters notification state, and closes its agent and slash worker.
    """
    sessions = getattr(mod, "_sessions", None)
    if not isinstance(sessions, dict):
        return
    for sid in list(sessions):
        mod._close_session_by_id(sid, end_reason="test_cleanup")


@pytest.fixture(autouse=True)
def _reset_tui_gateway_server_state():
    mod = sys.modules.get(_TUI_SERVER_MODULE)
    snapshot = None
    if mod is not None:
        snapshot = {
            "methods": dict(mod._methods),
            "cfg": (mod._cfg_cache, mod._cfg_mtime, mod._cfg_path),
            "db": (mod._db, mod._db_error),
            "real_stdout": mod._real_stdout,
        }

    yield

    mod = sys.modules.get(_TUI_SERVER_MODULE)
    if mod is None:
        return

    # This finalizer can run before the test's own monkeypatch undo, so a
    # global may still be replaced with a non-dict test double — skip those
    # (monkeypatch restores the real, pre-test object afterwards anyway).
    sessions = mod._sessions
    if isinstance(sessions, dict):
        _teardown_tui_server_sessions(mod)
    for name in (
        "_pending",
        "_pending_prompt_payloads",
        "_answers",
        "_child_mirrors",
        "_active_child_runs",
    ):
        obj = getattr(mod, name, None)
        if isinstance(obj, dict):
            obj.clear()

    if snapshot is not None:
        mod._methods.clear()
        mod._methods.update(snapshot["methods"])
        mod._cfg_cache, mod._cfg_mtime, mod._cfg_path = snapshot["cfg"]
        mod._db, mod._db_error = snapshot["db"]
        mod._real_stdout = snapshot["real_stdout"]
    else:
        # First imported during this test — reset to import-time defaults
        # for the globals we could not snapshot (``_methods`` is left to
        # the importing file's fixture, see block comment above).
        mod._cfg_cache = None
        mod._cfg_mtime = None
        mod._cfg_path = None
        mod._db = None
        mod._db_error = None

    # A leaked context-local Hermes home override redirects every later
    # ``get_hermes_home()`` call (active-session registry, config paths)
    # to a stale per-test tmpdir. Force the main-thread ContextVar back
    # to its default.
    try:
        from hermes_constants import get_hermes_home_override, set_hermes_home_override

        if get_hermes_home_override() is not None:
            set_hermes_home_override(None)
    except Exception:
        pass


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
#    ``check_output`` reject any ``systemctl ... <verb> hermes-gateway``
#    invocation that would mutate the live unit. Read-only systemctl
#    calls (``status``, ``show``, ``list-units``) still pass through.
#
# We intentionally do NOT stub ``find_gateway_pids`` / ``_scan_gateway_pids``
# here — tests of those functions themselves need the real implementation.
# Even if a test gets the live gateway PID back from a real scan, the
# ``os.kill`` guard above catches the actual signal call, and the
# ``systemctl`` guard catches the systemd path. Discovery without
# delivery is harmless.

_LIVE_SYSTEM_GUARD_BYPASS_MARK = "live_system_guard_bypass"
_REQUIRES_WAL_MARK = "requires_wal"


def _wal_is_usable() -> bool:
    """True when Hermes will actually put a database into WAL mode here.

    Hermes refuses journal_mode=WAL on SQLite builds carrying the upstream
    WAL-reset corruption bug (3.7.0–3.51.2, excluding backports 3.50.7 /
    3.44.6) and falls back to DELETE. On such a build NO ``-wal`` sidecar is
    ever created, so a test asserting on WAL frames, ``-wal`` file size, or
    checkpoint behaviour cannot pass — it is testing a mode the runtime
    declined to enable, not a regression.

    This matters because the interpreter running the tests and the interpreter
    running Hermes can link DIFFERENT SQLite versions: a repo ``.venv`` on
    3.50.4 (vulnerable → DELETE) alongside a Hermes managed runtime on 3.53.1
    (fixed → WAL). The same test then passes in one and fails in the other.

    IMPORTANT: this must NOT import ``hermes_state``. That module computes
    ``DEFAULT_DB_PATH`` from ``get_hermes_home()`` at import time, so importing
    it during collection — before the per-test ``_isolate_hermes_home`` fixture
    redirects ``HERMES_HOME`` — permanently caches the DEVELOPER'S REAL
    ``~/.hermes/state.db`` for the whole session. Tests then read live
    production sessions instead of a tempdir. The version predicate is
    duplicated from ``hermes_state._is_sqlite_wal_reset_vulnerable`` (upstream
    fixed ranges, stable) rather than imported, and
    ``test_conftest_wal_gate.py`` pins the two implementations in agreement.
    """
    info = sqlite3.sqlite_version_info
    if info < (3, 7, 0):
        return True  # pre-WAL library: cannot hit the race
    if info >= (3, 51, 3):
        return True  # fixed upstream
    if (3, 50, 7) <= info < (3, 51, 0):
        return True  # 3.50.x backport
    if (3, 44, 6) <= info < (3, 45, 0):
        return True  # 3.44.x backport
    return False


# ── Audio-playback guard ───────────────────────────────────────────────────
#
# Same class of incident as the live-system guard above, different primitive:
# a test run spoke the string "partial answer complete" out of the developer's
# speakers. That string is a test fixture
# (``tests/test_tui_gateway_server.py``'s fake ``final_response``), and the
# route it took is fully in-process — no leaked shell variable required:
#
#   1. ``test_voice_toggle_tts_branch_also_carries_record_key`` drives the
#      ``voice.toggle`` RPC with ``action="tts"``. The handler
#      (``tui_gateway/server.py``) flips the flag by writing the *real*
#      process environment: ``os.environ["HERMES_VOICE_TTS"] = "1"``. The
#      test's ``monkeypatch.delenv(..., raising=False)`` records no undo entry
#      (pytest only records an undo when the key was present), so the "1"
#      survives teardown and persists for the rest of the pytest process.
#   2. Any later test in that process that drives a turn to completion hits
#      the TTS dispatch in ``prompt.submit``, which checks
#      ``_voice_tts_enabled()`` — now true — and fires
#      ``hermes_cli.voice.speak_text(final_response)`` on a daemon thread.
#   3. ``speak_text`` needs no API key to be audible: ``tools/tts_tool.py``
#      defaults to the ``edge`` provider, which is keyless.
#
# Because the flag is set from *inside* the process, ``scripts/run_tests.sh``'s
# ``env -i`` does not help, and neither does env-blanking on its own — the
# hermetic fixture blanks at test setup, and step 1 re-sets it mid-test. So we
# also intercept the primitive that does the damage, exactly as the
# live-system guard intercepts ``os.kill`` rather than trusting every caller
# to mock it:
#
#  • ``hermes_cli.voice.speak_text`` — the synth+playback entry point both
#    gateway call sites late-import, so patching the module attribute catches
#    them wherever they import it from.
#  • ``hermes_cli.voice.play_audio_file`` — the module-level binding
#    ``speak_text`` actually plays through. Patching the binding inside
#    ``hermes_cli.voice`` (not ``tools.voice_mode``) keeps the real function
#    available to the tests that legitimately exercise it with a mocked
#    audio backend (``tests/tools/test_voice_mode.py``).
#
# Config cannot re-open this hole: the ``tts:`` section of ``config.yaml``
# only selects *which* provider speaks, never *whether* to speak — that gate
# is the env var alone.

_AUDIO_GUARD_BYPASS_MARK = "real_audio_playback"
_ALLOW_MACOS_KEYCHAIN_MARK = "allow_macos_keychain"

# ---------------------------------------------------------------------------
# OS gating
#
# Hermes runs on Linux, macOS and native Windows, and a lot of its behaviour
# genuinely differs per host: PTY vs pywinpty, taskkill vs SIGTERM, launchd
# vs systemd, Keychain vs libsecret, ``%LOCALAPPDATA%`` vs ``~/.hermes``.
#
# Historically those code paths were tested by *faking* the host — patching
# ``sys.platform`` to ``"win32"`` inside a Linux CI job. That gives a green
# test on a machine where the code under test could not actually run: the
# fake covers the ``if sys.platform == "win32"`` branch selection but nothing
# underneath it (``msvcrt`` still isn't importable, ``taskkill`` still isn't
# on PATH, paths are still POSIX, ``signal.SIGKILL`` still exists). The
# result was tests that pass on Linux and tell us nothing about Windows.
#
# So: a test whose subject is genuinely OS-specific declares the OS it
# belongs to and runs there for real —
#
#   @pytest.mark.windows_only   → only on native Windows (``sys.platform == "win32"``)
#   @pytest.mark.macos_only     → only on macOS (``sys.platform == "darwin"``)
#   @pytest.mark.linux_only     → only on Linux (``sys.platform.startswith("linux")``)
#
# Elsewhere the test is skipped, not faked. CI runs a dedicated macOS job
# (``-m macos_only``) and a dedicated Windows job (``-m windows_only``) so
# those markers are actually exercised on their own host rather than
# quietly skipped everywhere.
#
# This does NOT mean every mention of another platform must be gated. Two
# things are legitimately host-independent and stay on the Linux runner:
#
#   • Pure functions that TAKE a platform as data — e.g.
#     ``hidden_windows_child_options(opts, is_windows=True)`` or a
#     ``resolve_launcher(platform_name)`` helper. Passing "win32" as an
#     argument is not faking the host; the function's whole contract is
#     that it maps input to output.
#   • Declaration/packaging invariants — e.g. "pyproject declares tzdata
#     with a ``sys_platform == 'win32'`` marker". That's an assertion about
#     a file, not about runtime behaviour.
#
# The line is: if the test needs the interpreter to BELIEVE it is on
# another OS in order to pass, it belongs on that OS.
# ---------------------------------------------------------------------------

_OS_MARKS = {
    "linux_only": (
        lambda: sys.platform.startswith("linux"),
        "Linux",
    ),
    "macos_only": (
        lambda: sys.platform == "darwin",
        "macOS",
    ),
    "windows_only": (
        lambda: sys.platform == "win32",
        "native Windows",
    ),
}


def pytest_configure(config):  # noqa: D401 — pytest hook
    """Register markers used by hermetic conftest."""
    config.addinivalue_line(
        "markers",
        f"{_LIVE_SYSTEM_GUARD_BYPASS_MARK}: bypass the live-system guard "
        "(only for tests that genuinely need real os.kill / subprocess "
        "behaviour — e.g. PTY tests that signal their own child).",
    )
    config.addinivalue_line(
        "markers",
        f"{_REQUIRES_WAL_MARK}: test needs the runtime to actually enable "
        "SQLite WAL mode; skipped on builds where Hermes falls back to "
        "journal_mode=DELETE for the WAL-reset bug.",
    )
    config.addinivalue_line(
        "markers",
        f"{_AUDIO_GUARD_BYPASS_MARK}: bypass the audio-playback guard (only "
        "for tests that genuinely need real TTS synthesis and speaker "
        "playback — there are none in the default suite).",
    )
    config.addinivalue_line(
        "markers",
        f"{_ALLOW_MACOS_KEYCHAIN_MARK}: allow a test to exercise the macOS "
        "Keychain credential reader with its own subprocess/platform mocks.",
    )
    config.addinivalue_line(
        "markers",
        "require_symlinks: skip the test if symbolic links cannot be "
        "created in the current environment (needs admin/developer mode "
        "on Windows).",
    )
    config.addinivalue_line(
        "markers",
        "real_memory_guard: bypass the autouse fixture that pins the kanban "
        "dispatcher's memory guard to 'no data' — only for tests that "
        "exercise the guard itself with their own patched samples.",
    )
    # NOTE: linux_only / macos_only / windows_only are declared in
    # pyproject.toml's ``markers`` list, not here — they are part of the
    # project's public marker vocabulary (``pytest --markers``, and the CI
    # lanes select on them), whereas the marks above are conftest-internal
    # guards. Declaring them in both places just meant two descriptions that
    # could drift apart.

    # The pyproject addopts pin ``--timeout-method=signal`` relies on
    # ``signal.SIGALRM``, which does not exist on Windows — pytest-timeout
    # raises AttributeError at timer setup and the whole run aborts before any
    # test executes. Fall back to the thread-based timer on Windows so the
    # suite runs natively there (POSIX keeps the more reliable signal method).
    if sys.platform == "win32" and getattr(config.option, "timeout_method", None) == "signal":
        config.option.timeout_method = "thread"


_symlink_supported_cache = None


def _check_symlink_support() -> bool:
    global _symlink_supported_cache
    if _symlink_supported_cache is not None:
        return _symlink_supported_cache

    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.touch()
            lnk = Path(d) / "lnk"
            lnk.symlink_to(src)
            _symlink_supported_cache = True
            return True
    except OSError:
        _symlink_supported_cache = False
        return False


def pytest_runtest_setup(item):
    if item.get_closest_marker("require_symlinks"):
        if not _check_symlink_support():
            pytest.skip(
                "Environment does not support symbolic links "
                "(requires admin/developer mode on Windows)"
            )


def _reject_multiple_os_marks(items):
    """Fail collection when one test carries two host-OS markers.

    Every marker in ``_OS_MARKS`` skips on all but one host, so two of them
    on the same item means it is skipped on *every* host — a test that never
    runs anywhere, reported as green by both the Linux suite and the
    tests-os lanes. That is the exact silent-coverage-loss the markers were
    introduced to remove, so it is a hard collection error rather than a
    warning nobody reads.
    """
    offenders = []
    for item in items:
        marks = sorted({m.name for m in item.iter_markers() if m.name in _OS_MARKS})
        if len(marks) > 1:
            offenders.append(f"  {item.nodeid}: {', '.join(marks)}")
    if offenders:
        raise pytest.UsageError(
            "a test may carry at most one host-OS marker "
            f"({', '.join(_OS_MARKS)}); these carry several and would be "
            "skipped on every host:\n" + "\n".join(offenders)
        )


def pytest_collection_modifyitems(config, items):  # noqa: D401 — pytest hook
    """Apply host-OS gating, then skip ``requires_wal`` where WAL is unusable.

    OS gating: a test marked ``linux_only`` / ``macos_only`` /
    ``windows_only`` runs only on that host. See the ``_OS_MARKS`` block
    comment above for why these tests are skipped rather than run against a
    patched ``sys.platform``.

    WAL gating is cheaper and more honest than each test hand-rolling a
    version check: the reason string names the actual linked version so the
    skip is diagnosable rather than mysterious.
    """
    _reject_multiple_os_marks(items)

    for mark_name, (is_host, label) in _OS_MARKS.items():
        if is_host():
            continue
        skip_os = pytest.mark.skip(
            reason=f"{label}-only test (marked {mark_name}); host is {sys.platform}"
        )
        for item in items:
            if item.get_closest_marker(mark_name) is not None:
                item.add_marker(skip_os)

    if _wal_is_usable():
        return

    reason = (
        f"SQLite {sqlite3.sqlite_version} has the WAL-reset bug — Hermes uses "
        "journal_mode=DELETE here, so no -wal sidecar exists to assert on"
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker(_REQUIRES_WAL_MARK) is not None:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True)
def _live_system_guard(request, monkeypatch):
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
    import shlex as _shlex
    import subprocess as _subprocess

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
            if int(pgid) == own_pgid or _is_own_subtree(int(pgid)):
                return real_killpg(pgid, sig, *args, **kwargs)
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"os.killpg({pgid}, {sig}) — PGID is outside the test "
                "process group. See _live_system_guard for the why."
            )

        monkeypatch.setattr(_os, "killpg", _guarded_killpg)

    # ── Subprocess command-string inspection (whole-line) ──────────
    _HERMES_TOKENS = (
        "hermes-gateway",
        "hermes.service",
        "hermes_cli.main gateway",
        "hermes_cli/main.py gateway",
        "gateway/run.py",
        "hermes gateway",
    )
    _MUTATING_VERBS = (
        "restart", "start", "stop", "kill", "reload",
        "reset-failed", "enable", "disable", "mask", "unmask",
        "daemon-reload", "try-restart", "reload-or-restart",
    )
    _PROCESS_KILLERS = ("pkill", "killall", "taskkill", "skill", "fuser")
    # Shell/launcher executables whose arguments are themselves commands —
    # argv[0]-only scanning must not exempt what they wrap.
    _WRAPPER_COMMANDS = (
        "sh", "bash", "zsh", "dash", "env", "nohup", "setsid",
        "timeout", "sudo", "xargs", "nice", "ionice", "stdbuf", "flock",
    )

    def _cmd_to_string(cmd) -> str:
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
                return " ".join(str(t) for t in cmd)
            except Exception:
                return ""
        return str(cmd)

    def _matches_hermes_gateway(cmd_str: str) -> bool:
        low = cmd_str.lower()
        return any(tok in low for tok in _HERMES_TOKENS)

    def _is_blocked_systemctl(cmd) -> bool:
        cmd_str = _cmd_to_string(cmd)
        if "systemctl" not in cmd_str:
            return False
        if not _matches_hermes_gateway(cmd_str):
            return False
        try:
            tokens = _shlex.split(cmd_str)
        except ValueError:
            tokens = cmd_str.split()
        return any(verb in tokens for verb in _MUTATING_VERBS)

    def _is_process_killer(cmd) -> bool:
        cmd_str = _cmd_to_string(cmd)
        try:
            tokens = _shlex.split(cmd_str)
        except ValueError:
            tokens = cmd_str.split()
        if not tokens:
            return False

        # For argv-style calls only argv[0] is the executable; scanning every
        # argument blocked innocent commands like ``cat /tmp/.../skill``
        # ("skill" is in _PROCESS_KILLERS).  Wrapper executables still get
        # full-token scanning so ``["bash", "-c", "pkill ..."]`` stays caught.
        if isinstance(cmd, (list, tuple)):
            head0 = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            killer_tokens = tokens if head0 in _WRAPPER_COMMANDS else tokens[:1]
        else:
            killer_tokens = tokens
        for tok in killer_tokens:
            head = tok.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if head in _PROCESS_KILLERS:
                low = cmd_str.lower()
                # pkill -f pattern: catch hermes-themed patterns + a
                # plain "python" -f which would catch the live gateway
                # whose cmdline contains "python -m hermes_cli.main".
                if (
                    "hermes" in low
                    or "gateway" in low
                    or ("python" in low and "-f" in tokens)
                ):
                    return True
        return False

    def _check_subprocess_cmd(name, cmd):
        if _is_blocked_systemctl(cmd):
            raise RuntimeError(
                f"tests/conftest.py live-system guard: blocked "
                f"subprocess.{name}({cmd!r}) — would mutate the "
                "live hermes-gateway systemd unit. Mock "
                "subprocess.run / _run_systemctl in the test, or "
                "mark with @pytest.mark.live_system_guard_bypass."
            )
        if _is_process_killer(cmd):
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
        cmd_str = _cmd_to_string(cmd)
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
            _check_subprocess_cmd(name, cmd)
            return real(cmd, *args, **kwargs)
        _guarded.__name__ = f"_guarded_{name}"
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


@pytest.fixture(autouse=True)
def _audio_playback_guard(request, monkeypatch):
    """Stub TTS synthesis + speaker playback for every test.

    See the block comment above for the incident this closes. Defence in
    depth behind ``_HERMES_BEHAVIORAL_VARS``: the env blanking stops the flag
    leaking *between* tests, this stops the speakers ever opening even when a
    test sets the flag *itself* (which the ``voice.toggle`` RPC handler does,
    by writing ``os.environ`` directly).

    Deliberately silent rather than raising: unlike a stray ``os.kill``, a
    stray ``speak_text`` is dispatched on a daemon thread whose exception
    nobody would ever see, so a hard failure would neither stop the test nor
    surface. Silence is the whole point. Tests that genuinely want real audio
    can opt out with ``@pytest.mark.real_audio_playback``.
    """
    if request.node.get_closest_marker(_AUDIO_GUARD_BYPASS_MARK):
        yield
        return

    try:
        import hermes_cli.voice as _voice
    except Exception:
        # Optional audio deps missing — nothing importable to speak with.
        yield
        return

    def _blocked_speak_text(text, *args, **kwargs):
        return None

    def _blocked_play_audio_file(path, *args, **kwargs):
        return False

    if hasattr(_voice, "speak_text"):
        monkeypatch.setattr(_voice, "speak_text", _blocked_speak_text)
    if hasattr(_voice, "play_audio_file"):
        monkeypatch.setattr(_voice, "play_audio_file", _blocked_play_audio_file)

    yield


@pytest.fixture(autouse=True)
def _isolate_computer_use_approval_state():
    """Reset computer-use approval globals after every test.

    ``tools.computer_use.tool`` keeps three module-globals for the CLI
    approval flow: ``_approval_callback`` (set by the CLI console on init)
    plus the per-session unlock stores ``_always_allow`` /
    ``_session_auto_approve``. A test that installs a callback — or drives
    CLI init far enough that the real one is registered — and does not reset
    it poisons every later computer-use test in the same process:

    * a leaked callback that raises (dead UI/queue infra, or a stale
      two-argument signature — the real contract is ``(action, args,
      summary)``) turns into ``verdict = "deny"`` in ``_request_approval``,
      so dispatch tests fail with an empty backend call list;
    * a leaked callback that blocks (the real CLI one waits on an answer
      queue) hangs the whole single-process run forever — pytest-timeout is
      the only thing that can cut it.

    Both symptoms are order-dependent: the affected files pass in isolation
    and only fail in full-suite runs. Teardown-only, so tests that install
    their own callback keep it for their own duration.
    """
    yield
    try:
        from tools.computer_use import tool as _cu_tool

        _cu_tool.set_approval_callback(None)
        with _cu_tool._approval_lock:
            _cu_tool._always_allow.clear()
            _cu_tool._session_auto_approve.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _moa_caches_isolated():
    """Clear module-level MoA cold-start caches before each test.

    ``agent.moa_loop`` caches the resolved preset and each slot's provider
    runtime at module level (keyed on config mtime / provider+model) so the
    tool loop doesn't re-resolve them serially on every iteration. Tests
    monkeypatch resolvers and config paths, so a cache entry leaked from one
    test would poison the next. Clear both around every test.
    """
    import agent.moa_loop as moa

    moa._preset_cache.clear()
    moa._runtime_cache.clear()
    yield
    moa._preset_cache.clear()
    moa._runtime_cache.clear()

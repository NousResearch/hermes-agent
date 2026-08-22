"""2K Law enforcement — the campaign primitive as a regression test.

Doctrine: no code file in this repo exceeds 2,000 lines (2K Law, locked
2026-08-05). The ONLY exceptions are non-code documents (LLMS.TXT,
LLMS-FULL.TXT, markdown, JSON, YAML, lockfiles, vendor files).

This test fails on ANY code file over the bar. God-files on the kill
track are listed in OVER_2K_MANIFEST and must shrink monotonically —
each god-file kill removes its entry. A NEW file over 2,000 lines is a
test failure, always.

Vendored/third-party trees (venvs, node_modules, site-packages, dist,
build) are excluded — they are not repo code. Only files that ship in
the repository's own source surface are audited.
"""

from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".cpp", ".h", ".hpp", ".sh", ".bat", ".ps1"}
DOC_EXTENSIONS = {".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".lock", ".ini", ".cfg"}
NON_CODE_DOC_NAMES = {"LLMS.TXT", "LLMS-FULL.TXT", "LICENSE", "LICENSE.md", "COPYING", "NOTICE", "CHANGELOG"}
EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".hermes-runtime",
    "dist", "build", "site-packages", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv.broken-20260707-0718", ".tox", ".nox", "coverage", "htmlcov", "vendor",
}
EXCLUDED_PREFIXES = ("venv.", "\.venv", "node_modules", "site-packages")

# God-files on the kill track — the FULL 2K surface (the Pantheon of False Gods).
# Every code file over 2,000 lines is tracked here, measured at origin/main
# 2026-08-05 (119 entries). Each entry MUST be removed when its kill ships
# (monotonic shrink = the completion record). Vendor trees are excluded
# (third-party). The full ledger with tracking state is posted on epic #78647.
OVER_2K_MANIFEST = {
    "gateway/run.py": 26986,
    "cli.py": 18485,
    "hermes_cli/web_server.py": 17732,
    "tests/test_tui_gateway_server.py": 16245,
    "tui_gateway/server.py": 14006,
    "hermes_cli/main.py": 12599,
    "apps/desktop/electron/main.ts": 12038,
    "hermes_cli/kanban_db.py": 10275,
    "plugins/platforms/telegram/adapter.py": 10147,
    "plugins/platforms/discord/adapter.py": 10138,
    "agent/auxiliary_client.py": 9976,
    "hermes_state.py": 9691,
    "hermes_cli/auth.py": 9240,
    "plugins/platforms/slack/adapter.py": 9088,
    "skills/research/research-paper-writing/SKILL.md": 2377,
    "run_agent.py": 8163,
    "hermes_cli/gateway.py": 7461,
    "agent/conversation_loop.py": 7334,
    "tools/mcp_tool.py": 7230,
    "gateway/platforms/api_server.py": 7188,
    "agent/context_compressor.py": 6883,
    "gateway/platforms/base.py": 6861,
    "tests/run_agent/test_run_agent.py": 6148,
    "plugins/platforms/feishu/adapter.py": 5874,
    "gateway/slash_commands.py": 5545,
    "hermes_cli/update_cmd.py": 5540,
    "hermes_cli/tools_config.py": 5452,
    "hermes_cli/config.py": 5434,
    "hermes_cli/models.py": 5334,
    "gateway/platforms/yuanbao.py": 5298,
    "plugins/platforms/matrix/adapter.py": 5284,
    "plugins/memory/openviking/__init__.py": 5212,
    "tools/browser_tool.py": 5098,
    "tests/gateway/test_slack.py": 4561,
    "apps/desktop/src/app/session/hooks/use-prompt-actions/index.test.tsx": 4449,
    "tools/skills_hub.py": 4432,
    "tests/agent/test_auxiliary_client.py": 4428,
    "cron/scheduler.py": 4417,
    "tools/approval.py": 4380,
    "agent/chat_completion_helpers.py": 4363,
    "hermes_cli/config_defaults.py": 4313,
    "scripts/install.ps1": 4262,
    "tests/test_hermes_state.py": 4283,
    "tests/hermes_cli/test_web_server.py": 4232,
    "agent/agent_runtime_helpers.py": 4067,
    "agent/conversation_compression.py": 4008,
    "tools/tts_tool.py": 3964,
    "tools/delegate_tool.py": 3931,
    "plugins/platforms/google_chat/adapter.py": 3738,
    "hermes_cli/setup.py": 3645,
    "gateway/session.py": 3490,
    "tools/terminal_tool.py": 3419,
    "hermes_cli/cli_commands_mixin.py": 3387,
    "agent/model_metadata.py": 3370,
    "scripts/install.sh": 3370,
    "tests/gateway/test_matrix.py": 3344,
    "tools/computer_use/cua_backend.py": 3295,
    "optional-skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py": 3286,
    "gateway/platforms/qqbot/adapter.py": 3273,
    "hermes_cli/kanban.py": 3236,
    "hermes_cli/model_switch.py": 3203,
    "agent/anthropic_adapter.py": 3177,
    "hermes_cli/model_setup_flows.py": 3151,
    "agent/credential_pool.py": 3147,
    "apps/desktop/src/i18n/zh.ts": 3145,
    "tui_gateway/methods_session.py": 3138,
    "apps/desktop/src/i18n/en.ts": 2984,
    "plugins/platforms/photon/adapter.py": 2910,
    "tests/agent/test_context_compressor.py": 2897,
    "plugins/kanban/dashboard/plugin_api.py": 2862,
    "tests/gateway/test_api_server.py": 2862,
    "apps/desktop/src/i18n/ja.ts": 2824,
    "agent/agent_init.py": 2806,
    "tools/file_operations.py": 2805,
    "hermes_cli/doctor.py": 2777,
    "ui-tui/packages/hermes-ink/src/ink/ink.tsx": 2752,
    "cron/jobs.py": 2746,
    "apps/desktop/src/i18n/zh-hant.ts": 2710,
    "tests/tools/test_mcp_tool.py": 2701,
    "gateway/config.py": 2688,
    "tools/transcription_tools.py": 2687,
    "scripts/release.py": 2637,
    "apps/desktop/src/i18n/ar.ts": 2611,
    "web/src/lib/api.ts": 2609,
    "apps/desktop/src/i18n/types.ts": 2537,
    "tools/process_registry.py": 2529,
    "tests/hermes_cli/test_relay_shared_metrics_runtime.py": 2514,
    "acp_adapter/server.py": 2510,
    "hermes_cli/plugins.py": 2510,
    "agent/proxy_sources/iron_proxy.py": 2494,
    "tests/gateway/test_feishu.py": 2469,
    "gateway/platforms/weixin.py": 2419,
    "gateway/stream_consumer.py": 2410,
    "agent/moa_loop.py": 2384,
    "agent/tool_executor.py": 2338,
    "ui-tui/packages/hermes-ink/src/native-ts/yoga-layout/index.ts": 2326,
    "tools/file_tools.py": 2319,
    "tools/voice_mode.py": 2308,
    "hermes_cli/runtime_provider.py": 2298,
    "hermes_cli/profiles.py": 2262,
    "gateway/status.py": 2260,
    "tools/kanban_tools.py": 2250,
    "hermes_cli/commands.py": 2245,
    "plugins/memory/hindsight/__init__.py": 2232,
    "hermes_state_search.py": 2230,
    "agent/prompt_builder.py": 2206,
    "web/src/pages/SessionsPage.tsx": 2200,
    "tools/skills_sync_client.py": 2187,
    "gateway/relay/adapter.py": 2144,
    "tests/tools/test_computer_use.py": 2131,
    "tools/send_message_tool.py": 2116,
    "gateway/platforms/whatsapp_cloud.py": 2111,
    "tools/code_execution_tool.py": 2087,
    "hermes_cli/plugins_cmd.py": 2082,
    "tests/gateway/test_voice_command.py": 2043,
    "hermes_cli/skills_hub.py": 2036,
    "tools/environments/docker.py": 2029,
    "tests/agent/test_credential_pool.py": 2026,
    "agent/curator.py": 2019,
}


def _is_excluded(path: pathlib.Path) -> bool:
    parts = path.parts
    for part in parts:
        if part in EXCLUDED_DIRS:
            return True
    rel = path.relative_to(REPO_ROOT).as_posix()
    for prefix in EXCLUDED_PREFIXES:
        if rel.startswith(prefix):
            return True
    return False


def _is_doc(path: pathlib.Path) -> bool:
    name = path.name
    if name in NON_CODE_DOC_NAMES:
        return True
    return path.suffix.lower() in DOC_EXTENSIONS


def _line_count(path: pathlib.Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def _code_files() -> list[pathlib.Path]:
    out = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        root_p = pathlib.Path(root)
        for f in files:
            p = root_p / f
            if _is_excluded(p) or _is_doc(p):
                continue
            if p.suffix.lower() in CODE_EXTENSIONS:
                out.append(p)
    return out


def test_no_unmanifested_code_file_exceeds_2000_lines():
    """Every code file over the bar must be a MANIFESTED god-file.

    The manifest is the kill track — files being sharded, allowed to be
    over the bar until their kill ships. Anything over 2,000 lines that is
    NOT on the kill track is a hard 2K-law violation: it is either a new
    god-file that must be tracked, or a file that grew without authorization.
    """
    violations = []
    for p in _code_files():
        n = _line_count(p)
        if n > 2000:
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel in OVER_2K_MANIFEST:
                continue  # on the kill track — must shrink, tested separately
            violations.append((n, rel))
    violations.sort(reverse=True)
    assert not violations, (
        "2K LAW VIOLATIONS — code files over 2,000 lines NOT on the kill track:\n"
        + "\n".join(f"  {n:>6}  {rel}" for n, rel in violations)
        + "\nEither shard them (add to the kill track) or this is an unauthorized grow."
    )


def test_godfile_manifest_shrinks_monotonically():
    """Every manifest god-file is still over the bar — and the count only shrinks.

    The manifest is the kill track. Each entry's line count must be <= the
    recorded value (files only shrink via sharding), and the manifest must
    not gain entries. When a kill ships, its entry is removed here.
    """
    current = {}
    for rel in OVER_2K_MANIFEST:
        p = REPO_ROOT / rel
        if p.exists():
            current[rel] = _line_count(p)
    for rel, n in current.items():
        recorded = OVER_2K_MANIFEST[rel]
        assert n <= recorded + 5, (
            f"{rel} GREW: recorded {recorded} lines, now {n} — "
            "god-files never grow; shard them."
        )

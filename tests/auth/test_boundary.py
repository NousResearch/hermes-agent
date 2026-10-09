"""Prevent reverse CLI dependencies in the canonical auth package."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTH = ROOT / "auth"
FORBIDDEN = ("hermes_cli", "nous_cli")


def test_auth_has_no_cli_imports():
    assert (AUTH / "__init__.py").is_file()
    violations = []
    for path in sorted(AUTH.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == prefix or name.startswith(prefix + ".")
                       for prefix in FORBIDDEN):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "Reverse CLI dependency:\n" + "\n".join(violations)


def test_runtime_has_no_retired_pool_or_grant_imports():
    retired = {
        "agent.credential_sources", "agent.anthropic_credentials",
        "hermes_cli.auth_qwen",
        "agent.credential_pool", "agent.credential_pool_admin",
        "agent.credential_pool_model_cooldowns", "agent.credential_pool_plugin",
        "hermes_cli.auth_oauth_grants",
    }
    paths = list(ROOT.glob("*.py"))
    for directory in ("auth", "agent", "gateway", "hermes_cli", "tui_gateway",
                      "tools", "plugins", "cron", "acp_adapter"):
        paths.extend((ROOT / directory).rglob("*.py"))
    violations = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                if node.module in {"agent", "hermes_cli"}:
                    names.extend(node.module + "." + alias.name for alias in node.names)
            for name in names:
                if name in retired:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "Retired authentication dependency:\n" + "\n".join(violations)


def test_runtime_imports_authentication_operations_from_canonical_owners():
    """Routing metadata and presentation may remain at the application edge."""
    canonical = {
        "resolve_api_key_provider_secret": "auth.api_keys",
        "get_anthropic_key": "auth.api_keys",
        "get_codex_auth_status": "auth.provider_status",
        "get_xai_oauth_auth_status": "auth.provider_status",
        "get_minimax_oauth_auth_status": "auth.provider_status",
        "get_plugin_oauth_auth_status": "auth.provider_status",
        "_resolve_api_key_provider_secret": "auth.api_keys",
        "has_usable_secret": "auth.secret_validation",
        "looks_like_openrouter_key": "auth.secret_validation",
        "_usable_declared_secret": "auth.secret_validation",
        "is_rate_limited_auth_error": "auth.failure_policy",
        "strip_cloned_single_use_oauth_grants": "auth.oauth_grants",
        "heal_forked_single_use_oauth_grants": "auth.oauth_grants",
        "consume_oauth_heal_notices": "auth.oauth_grants",
        "resolve_nous_runtime_credentials": "auth.providers.nous",
        "resolve_codex_runtime_credentials": "auth.providers.codex",
        "resolve_xai_oauth_runtime_credentials": "auth.providers.xai",
        "resolve_qwen_runtime_credentials": "auth.providers.qwen",
        "resolve_minimax_oauth_runtime_credentials": "auth.providers.minimax",
        "resolve_spotify_runtime_credentials": "auth.providers.spotify",
        "get_spotify_auth_status": "auth.providers.spotify",
        "get_nous_auth_status_local": "auth.providers.nous_status",
    }
    retired = {"hermes_cli.nous_auth_keepalive", "hermes_cli.auth_qwen",
               "agent.anthropic_credentials", "agent.credential_pool"}
    paths = list(ROOT.glob("*.py"))
    for directory in ("agent", "gateway", "tools", "tui_gateway", "plugins",
                      "cron", "acp_adapter", "auth", "hermes_cli"):
        paths.extend((ROOT / directory).rglob("*.py"))
    violations = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in retired:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.module}")
                if node.module.startswith("hermes_cli.auth"):
                    for alias in node.names:
                        if alias.name in canonical:
                            violations.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}: "
                                f"{alias.name} belongs to {canonical[alias.name]}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in retired:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {alias.name}")
            # Lazy dispatch must follow the same ownership cut as static imports.
            elif isinstance(node, ast.Call) and len(node.args) >= 2:
                module, symbol = node.args[:2]
                if isinstance(module, ast.Constant) and isinstance(symbol, ast.Constant):
                    if module.value == "hermes_cli.auth" and symbol.value in canonical:
                        violations.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: lazy {symbol.value}")
    assert not violations, "CLI-owned runtime authentication:\n" + "\n".join(violations)


def test_authentication_presentation_does_not_implement_store_transactions():
    """CLI interaction consumes operations without owning auth.json writes."""
    forbidden = {"_auth_store_lock", "_save_auth_store", "_store_provider_state"}
    violations = []
    for path in sorted((ROOT / "hermes_cli").glob("auth*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if name in forbidden:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "CLI-owned store transaction:\n" + "\n".join(violations)

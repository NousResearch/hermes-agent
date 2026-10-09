"""Phase 5.2 live auth-projection contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli.auth_constants import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_PORTAL_URL,
    DEFAULT_NOUS_SCOPE,
    MINIMAX_OAUTH_CN_BASE,
    QWEN_OAUTH_CLIENT_ID,
    QWEN_OAUTH_TOKEN_URL,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_DEVICE_CODE_URL,
    XAI_OAUTH_SCOPE,
)
from hermes_cli.provider_auth import (
    AUTH_AUTO_DETECT_ORDER,
    get_provider_config,
    iter_auto_detect_provider_configs,
    iter_provider_configs,
)
from providers import get_provider_profile, list_providers

REPO = Path(__file__).resolve().parents[2]


def test_bundled_profiles_project_live_auth_configuration():
    profiles = {profile.name: profile for profile in list_providers()}
    configs = {config.id: config for config in iter_provider_configs()}

    assert configs.keys() == profiles.keys()
    for name in ("deepinfra", "anthropic", "bedrock", "copilot-acp"):
        profile = profiles[name]
        config = configs[name]
        assert config.name == profile.display_name
        assert config.auth_type == profile.auth_type
        assert config.inference_base_url == profile.base_url
        assert config.api_key_env_vars == tuple(profile.env_vars)
        assert config.base_url_env_var == profile.base_url_env_var


def test_alias_projects_the_effective_canonical_profile():
    profile = get_provider_profile("vercel")
    config = get_provider_config("vercel")

    assert profile is not None and config is not None
    assert profile.name == "ai-gateway"
    assert config.id == profile.name
    assert config.name == profile.display_name
    assert config.inference_base_url == profile.base_url


def test_auth_only_policy_is_projected_without_duplicating_provider_identity():
    nous = get_provider_config("nous")
    codex = get_provider_config("openai-codex")
    xai = get_provider_config("xai-oauth")
    qwen = get_provider_config("qwen-oauth")
    minimax = get_provider_config("minimax-oauth")

    assert nous is not None
    assert nous.portal_base_url == DEFAULT_NOUS_PORTAL_URL
    assert nous.client_id == DEFAULT_NOUS_CLIENT_ID
    assert nous.scope == DEFAULT_NOUS_SCOPE

    assert codex is not None
    assert codex.client_id == CODEX_OAUTH_CLIENT_ID
    assert codex.extra["token_url"] == CODEX_OAUTH_TOKEN_URL

    assert xai is not None
    assert xai.client_id == XAI_OAUTH_CLIENT_ID
    assert xai.scope == XAI_OAUTH_SCOPE
    assert xai.extra["device_code_url"] == XAI_OAUTH_DEVICE_CODE_URL

    assert qwen is not None
    assert qwen.client_id == QWEN_OAUTH_CLIENT_ID
    assert qwen.extra["token_url"] == QWEN_OAUTH_TOKEN_URL

    assert minimax is not None
    assert minimax.extra["cn_portal_base_url"] == MINIMAX_OAUTH_CN_BASE
    assert minimax.extra["cn_inference_base_url"] == get_provider_profile("minimax-cn").base_url


def test_auto_detect_order_is_policy_not_provider_declaration():
    assert AUTH_AUTO_DETECT_ORDER[:6] == (
        "openai-api",
        "gemini",
        "zai",
        "kimi-coding",
        "kimi-coding-cn",
        "stepfun",
    )

    detected = [config.id for config in iter_auto_detect_provider_configs()]
    legacy_detectable = [
        provider_id
        for provider_id in AUTH_AUTO_DETECT_ORDER
        if (config := get_provider_config(provider_id)) is not None
        and config.auth_type == "api_key"
        and config.api_key_env_vars
    ]
    assert [provider_id for provider_id in detected if provider_id in AUTH_AUTO_DETECT_ORDER] == legacy_detectable
    assert {"copilot", "lmstudio", "openrouter"}.isdisjoint(detected)


def _run_with_home(tmp_path: Path, plugin_dir_name: str, source: str, probe: str) -> dict:
    home = tmp_path / "home"
    plugin_dir = home / "plugins" / "model-providers" / plugin_dir_name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(source, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {plugin_dir_name}\nkind: model-provider\nversion: 0.0.1\ndescription: projection fixture\n",
        encoding="utf-8",
    )
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_user_home_plugin_projects_without_auth_registry_sync(tmp_path):
    result = _run_with_home(
        tmp_path,
        "projection-user",
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='projection-user', aliases=('projection-alias',),\n"
        "    display_name='Projection User', description='fixture', auth_type='api_key',\n"
        "    env_vars=('PROJECTION_KEY',), base_url_env_var='PROJECTION_ENDPOINT',\n"
        "    base_url='https://projection.example/v1'))\n",
        "import json\n"
        "from hermes_cli.provider_auth import get_provider_config\n"
        "cfg = get_provider_config('projection-alias')\n"
        "print(json.dumps({'id': cfg.id, 'name': cfg.name, 'auth': cfg.auth_type,\n"
        "    'env': list(cfg.api_key_env_vars), 'base_env': cfg.base_url_env_var,\n"
        "    'base': cfg.inference_base_url}))\n",
    )
    assert result == {
        "id": "projection-user",
        "name": "Projection User",
        "auth": "api_key",
        "env": ["PROJECTION_KEY"],
        "base_env": "PROJECTION_ENDPOINT",
        "base": "https://projection.example/v1",
    }


def test_user_home_override_projects_effective_profile_only(tmp_path):
    result = _run_with_home(
        tmp_path,
        "stepfun",
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(name='stepfun', aliases=('step',),\n"
        "    display_name='Step Override', description='fixture', auth_type='api_key',\n"
        "    env_vars=('STEP_OVERRIDE_KEY',), base_url_env_var='STEP_OVERRIDE_ENDPOINT',\n"
        "    base_url='https://override.example/v1'))\n",
        "import json\n"
        "from hermes_cli.provider_auth import get_provider_config\n"
        "cfg = get_provider_config('step')\n"
        "print(json.dumps({'id': cfg.id, 'name': cfg.name, 'env': list(cfg.api_key_env_vars),\n"
        "    'base_env': cfg.base_url_env_var, 'base': cfg.inference_base_url}))\n",
    )
    assert result == {
        "id": "stepfun",
        "name": "Step Override",
        "env": ["STEP_OVERRIDE_KEY"],
        "base_env": "STEP_OVERRIDE_ENDPOINT",
        "base": "https://override.example/v1",
    }

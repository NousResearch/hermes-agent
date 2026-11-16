"""Cold gateway config load imports only the platform adapters the profile configures.

``platform_registry.plugin_entries()`` materializes every deferred bundled adapter (discord,
telegram, google_chat SDKs, telegram's i18n catalog: ~0.5 s per cold start). The gateway config /
env passes must resolve only platforms with activation evidence (config block, env or .env under
the platform prefix / manifest ``requires_env``, credential pool), and a configured platform —
env-only included — must still be enabled.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PROBE = textwrap.dedent("""
    import json, sys
    from gateway.config import load_gateway_config
    from gateway.platform_registry import platform_registry
    config = load_gateway_config()
    print(json.dumps({
        "enabled": sorted(p.value for p, c in config.platforms.items() if c.enabled),
        "adapters": sorted(m.split("__", 1)[1].split(".")[0] for m in sys.modules
                           if m.startswith("hermes_plugins.platforms__") and m.endswith(".adapter")),
        "registered": sorted(platform_registry.registered_names()),
    }))
""")


def _load(tmp_path, *, config_yaml="", env=None, dotenv=""):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (home / ".env").write_text(dotenv, encoding="utf-8")
    # A clean child: the suite process has already imported adapters, and the developer's shell
    # may carry platform env vars that are genuine activation evidence.
    child_env = {k: v for k, v in os.environ.items()
                 if k in {"PATH", "LANG", "TZ", "HERMES_RUNTIME_DIR", "TMPDIR", "SYSTEMROOT"}}
    child_env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO), **(env or {}))
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(REPO), env=child_env,
                         capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_unconfigured_profile_imports_no_platform_adapter(tmp_path):
    result = _load(tmp_path)
    assert result["enabled"] == []
    assert result["adapters"] == [], f"unconfigured adapters imported at config load: {result['adapters']}"
    # Still registered (deferred): a later lookup by name imports it on demand.
    assert {"discord", "telegram", "google_chat", "slack"} <= set(result["registered"])


@pytest.mark.parametrize("case", ["env", "dotenv", "manifest_env", "yaml_sibling"])
def test_configured_platform_still_enabled_and_only_it_is_imported(tmp_path, case):
    kwargs, want, module = {
        "env": ({"env": {"TELEGRAM_BOT_TOKEN": "123456:fake-token-abcdef"}}, "telegram", "telegram"),
        "dotenv": ({"dotenv": "TELEGRAM_BOT_TOKEN=123456:fake-token-abcdef\n"}, "telegram", "telegram"),
        # sms is enabled by TWILIO_* (its manifest requires_env), not an SMS_ prefix.
        "manifest_env": ({"env": {"TWILIO_ACCOUNT_SID": "ACfake", "TWILIO_AUTH_TOKEN": "x",
                                  "TWILIO_PHONE_NUMBER": "+15550000000"}}, "sms", "sms"),
        # wecom_callback is registered by the wecom plugin.
        "yaml_sibling": ({"config_yaml": "platforms:\n  wecom_callback:\n    enabled: true\n    extra:\n"
                                         "      corp_id: c\n      corp_secret: s\n"}, "wecom_callback", "wecom"),
    }[case]
    result = _load(tmp_path, **kwargs)
    assert want in result["enabled"]
    assert result["adapters"] == [module], result

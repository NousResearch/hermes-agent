"""load_hermes_dotenv() reloads must not clobber values an external secret source resolved (#74265).

The gateway module import, per-turn reloads and cron fires all call ``load_hermes_dotenv()`` again;
``load_dotenv(override=True)`` writes the raw ``.env`` placeholder back and the once-per-home source
pass no longer re-applies, so without the restore the process authenticates with the placeholder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state():
    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    yield
    reg_module._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


def _register_fake_source(values: dict[str, str], *, override_existing: bool):
    """One bulk source supplying ``values``; ``override_existing`` mirrors the config knob the real
    sources expose (Bitwarden defaults True, the command source False)."""
    from agent.secret_sources import registry as reg_module
    from agent.secret_sources.base import FetchResult, SecretSource

    class _Fake(SecretSource):
        name = "fakebulk"
        label = "Fake"
        shape = "bulk"
        override_existing_default = override_existing

        def fetch(self, cfg, home_path):
            result = FetchResult()
            result.secrets = dict(values)
            return result

    reg_module.register_source(_Fake(), replace=True)


def _make_home(tmp_path: Path, monkeypatch, env_text: str, *, preserve: str = "") -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    secrets = "secrets:\n  fakebulk:\n    enabled: true\n"
    if preserve:
        secrets += f"  preserve_existing: [{preserve}]\n"
    (home / "config.yaml").write_text(secrets, encoding="utf-8")
    (home / ".env").write_text(env_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    return home


def test_source_secret_survives_second_load_hermes_dotenv(tmp_path, monkeypatch):
    """#74265: a source that overrides .env (Bitwarden's default) resolves the ``__BITWARDEN_MANAGED__``
    placeholder on load 1; load 2 (gateway import / per-turn reload / cron fire) must keep the
    resolved value instead of writing the placeholder back."""
    home = _make_home(tmp_path, monkeypatch, "GLM_API_KEY=__BITWARDEN_MANAGED__\n")
    # override_existing=True: the source is authoritative over .env, which is what makes the restore legal.
    _register_fake_source({"GLM_API_KEY": "vault-value"}, override_existing=True)

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["GLM_API_KEY"] == "vault-value"

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["GLM_API_KEY"] == "vault-value", (
        "second load_hermes_dotenv() wrote the .env placeholder back over the source-resolved value"
    )


@pytest.mark.parametrize(
    ("override_existing", "preserve"),
    [(False, ""), (True, "GLM_API_KEY")],
    ids=["gap-fill-source", "preserve_existing-name"],
)
def test_reload_restore_keeps_dotenv_precedence(tmp_path, monkeypatch, override_existing, preserve):
    """Names .env must win are never restored over a later .env edit: an ``override_existing: false``
    source only fills gaps, and a ``secrets.preserve_existing`` name keeps .env's value even against an
    overriding source. Both land in the per-home snapshot on load 1 (no .env value yet), so restoring
    the whole snapshot froze them at the source value; a cold start would have yielded ``local``."""
    home = _make_home(tmp_path, monkeypatch, "", preserve=preserve)
    _register_fake_source({"GLM_API_KEY": "vault-value"}, override_existing=override_existing)

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["GLM_API_KEY"] == "vault-value"

    (home / ".env").write_text("GLM_API_KEY=local\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["GLM_API_KEY"] == "local", (
        "reload restore re-asserted a source value over the user's .env edit"
    )

"""POST /api/model/set must never copy an env-expanded secret into config.yaml.

Regression for #88990: ``_apply_model_assignment_sync`` ran on the
env-EXPANDED config, so a provider entry whose raw yaml held
``api_key: ${MY_KEY}`` (or a ``key_env`` pointer) got its RESOLVED plaintext
key copied under ``model.api_key`` and persisted. ``_preserve_env_ref_templates``
cannot rescue it — no template ever lived at ``model.api_key``.

The fix prefers the credential pointer: ``key_env`` when the raw entry has
one, else the raw ``${VAR}`` template, and only falls back to the expanded
value when the key is stored as a literal on disk (no new exposure).
"""

import importlib
import sys

import pytest


SECRET = "sk-SUPERSECRET-e2e-12345"


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MY_SECRET_KEY", SECRET)
    # Config caches are keyed per-path, but reload the config module state so
    # nothing from a previous test's HERMES_HOME bleeds in.
    for mod in ("hermes_cli.config",):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    return home


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _apply(provider="myprov", model="test-model"):
    from hermes_cli.web_server import _apply_model_assignment_sync

    return _apply_model_assignment_sync("main", provider, model, "", "")


def _model_block(home) -> str:
    text = (home / "config.yaml").read_text(encoding="utf-8")
    return text.split("providers:")[0]


class TestModelSetSecretHandling:
    def test_env_template_key_is_copied_as_template_not_plaintext(self, isolated_home):
        _write_config(
            isolated_home,
            "model:\n  provider: openrouter\n  default: some/model\n"
            "providers:\n  myprov:\n    base_url: https://api.example.com/v1\n"
            "    api_key: ${MY_SECRET_KEY}\n    model: test-model\n",
        )

        _apply()

        text = (isolated_home / "config.yaml").read_text(encoding="utf-8")
        assert SECRET not in text
        assert "api_key: ${MY_SECRET_KEY}" in _model_block(isolated_home)

    def test_key_env_pointer_is_preferred_over_api_key(self, isolated_home, monkeypatch):
        monkeypatch.setenv("MYPROV_KEY", SECRET)
        _write_config(
            isolated_home,
            "model:\n  provider: openrouter\n  default: some/model\n"
            "providers:\n  myprov:\n    base_url: https://api.example.com/v1\n"
            "    key_env: MYPROV_KEY\n    api_key: ${MYPROV_KEY}\n    model: test-model\n",
        )

        _apply()

        text = (isolated_home / "config.yaml").read_text(encoding="utf-8")
        block = _model_block(isolated_home)
        assert SECRET not in text
        assert "key_env: MYPROV_KEY" in block
        assert "api_key" not in block

    def test_literal_on_disk_key_keeps_legacy_copy_behavior(self, isolated_home):
        _write_config(
            isolated_home,
            "model:\n  provider: openrouter\n  default: some/model\n"
            "providers:\n  myprov:\n    base_url: https://api.example.com/v1\n"
            "    api_key: sk-literal-on-disk\n    model: test-model\n",
        )

        _apply()

        assert "api_key: sk-literal-on-disk" in _model_block(isolated_home)

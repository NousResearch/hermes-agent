"""`hermes dump` / `hermes debug share` must not emit inline fallback keys.

``_config_overrides`` used to serialize the raw ``fallback_providers`` list
verbatim (``str(fallbacks)``), so an inline ``api_key`` landed in the dump —
which the bug-report template tells users to paste publicly and which
``hermes debug share`` uploads to a public paste service. Redaction on the
upload path never covered the dump text, and the downstream text redactor
misses custom-endpoint keys that carry no vendor prefix. The fix masks inline
credential values at the source, where the field names are known.
"""

from types import SimpleNamespace

from hermes_cli import dump

# A vendor-prefixed key and a prefix-less custom-endpoint key. The second is
# the case a pattern-based text redactor cannot catch — it only works because
# we mask by field name at the source.
_SK_KEY = "sk-live-abcdef0123456789-DO-NOT-SHIP"
_CUSTOM_KEY = "Zx9Qw3Rt7Kp2Mn5Bv8Lc4Hs1Df6Gj0"


def _fallback_entry(**extra):
    entry = {
        "name": "my-backup",
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": "https://api.example.com/v1",
    }
    entry.update(extra)
    return entry


def test_config_overrides_masks_inline_api_key():
    out = dump._config_overrides(
        {"fallback_providers": [_fallback_entry(api_key=_SK_KEY)]}
    )
    rendered = out["fallback_providers"]
    assert _SK_KEY not in rendered
    # masked form keeps head/tail only, per the dump's own convention
    assert "sk-l" in rendered and "SHIP" in rendered
    # non-secret structure is preserved so the dump stays useful
    assert "my-backup" in rendered and "https://api.example.com/v1" in rendered


def test_config_overrides_masks_prefixless_custom_key():
    # The downstream text redactor misses this shape; source masking catches it.
    out = dump._config_overrides(
        {"fallback_providers": [_fallback_entry(provider="custom", api_key=_CUSTOM_KEY)]}
    )
    assert _CUSTOM_KEY not in out["fallback_providers"]


def test_config_overrides_preserves_key_env_name():
    # key_env names an environment variable, not a secret — it must stay intact.
    out = dump._config_overrides(
        {"fallback_providers": [_fallback_entry(key_env="MY_PROVIDER_KEY")]}
    )
    assert "MY_PROVIDER_KEY" in out["fallback_providers"]


def test_config_overrides_masks_sibling_secret_fields():
    out = dump._config_overrides(
        {"fallback_providers": [_fallback_entry(token="tok-secret-value-1234567890")]}
    )
    assert "tok-secret-value-1234567890" not in out["fallback_providers"]


def test_dump_output_never_contains_raw_fallback_key(monkeypatch, capsys, tmp_path):
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model: gpt-4o\n"
        "provider: openai\n"
        "fallback_providers:\n"
        "  - name: my-backup\n"
        "    provider: openai\n"
        "    model: gpt-4o\n"
        "    base_url: https://api.example.com/v1\n"
        f"    api_key: {_SK_KEY}\n",
        encoding="utf-8",
    )

    dump.run_dump(SimpleNamespace(show_keys=False))
    out = capsys.readouterr().out

    # Only a real pass: the block rendered (masked marker present) AND the raw
    # key is absent. If the block never rendered, the masked marker is missing
    # and this fails rather than passing vacuously.
    assert "fallback_providers" in out
    assert "sk-l...SHIP" in out
    assert _SK_KEY not in out

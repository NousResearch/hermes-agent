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


# --- shapes that reach the same public paste by a different route -----------
#
# Masking the documented list-of-dicts shape is not enough: everything the
# helper does not explicitly recognize still goes through ``str()``, and
# ``str()`` on an arbitrary object runs its ``__repr__``. Each test below is
# a config that works at runtime and used to print its credential in full.


def test_masks_single_mapping_not_wrapped_in_a_list():
    # ``fallback_config._iter_fallback_entries`` accepts a bare mapping as a
    # one-entry chain, and ``resolve_entry_api_key`` reads its inline key — so
    # this config routes traffic. It must not skip masking for want of a list.
    out = dump._config_overrides(
        {"fallback_providers": _fallback_entry(api_key=_SK_KEY)}
    )
    assert _SK_KEY not in out["fallback_providers"]


def test_masks_secret_fields_outside_the_canonical_name():
    # Sibling credential names must be caught by the same rule as ``api_key``.
    for field in ("access_key", "auth_token", "client_secret", "passwd"):
        out = dump._config_overrides(
            {"fallback_providers": [_fallback_entry(**{field: _CUSTOM_KEY})]}
        )
        assert _CUSTOM_KEY not in out["fallback_providers"], field


def test_masks_non_string_values_under_a_secret_field():
    # A nested mapping and a bytes value both stringify to the key in full.
    for value in ({"value": _CUSTOM_KEY}, _CUSTOM_KEY.encode()):
        out = dump._config_overrides(
            {"fallback_providers": [_fallback_entry(api_key=value)]}
        )
        assert _CUSTOM_KEY not in out["fallback_providers"], type(value).__name__


def test_omits_entries_that_are_not_mappings():
    # A stray scalar in the list is not a usable entry, and its contents cannot
    # be verified — it is described by type rather than printed.
    out = dump._config_overrides(
        {"fallback_providers": [f"openai:{_SK_KEY}"]}
    )
    assert _SK_KEY not in out["fallback_providers"]
    assert "<omitted: str>" in out["fallback_providers"]


def test_omits_objects_whose_repr_would_print_the_key():
    class _Provider:
        def __repr__(self):
            return f"_Provider(api_key={_SK_KEY!r})"

    out = dump._config_overrides({"fallback_providers": [_Provider()]})
    assert _SK_KEY not in out["fallback_providers"]
    assert "<omitted: _Provider>" in out["fallback_providers"]


def test_strips_credentials_embedded_in_the_endpoint_url():
    # base_url stays visible — it is the field operators actually need — but
    # userinfo and secret query parameters are cleaned out of it.
    out = dump._config_overrides(
        {
            "fallback_providers": [
                _fallback_entry(base_url=f"https://user:{_SK_KEY}@api.example.com/v1")
            ]
        }
    )
    rendered = out["fallback_providers"]
    assert _SK_KEY not in rendered
    assert "api.example.com" in rendered

    out = dump._config_overrides(
        {
            "fallback_providers": [
                _fallback_entry(base_url=f"https://api.example.com/v1?api_key={_SK_KEY}")
            ]
        }
    )
    assert _SK_KEY not in out["fallback_providers"]
    assert "api.example.com" in out["fallback_providers"]


def test_masks_query_credentials_named_by_the_repository_policy():
    # The field-name markers ("key", "token", "secret"...) answer a question
    # about config fields and miss query names the repository already treats as
    # sensitive in agent/redact.py: a pre-signed URL signature and an OAuth
    # code carry no marker word at all.
    for param in ("signature", "x-amz-signature", "code", "access-token", "jwt"):
        out = dump._config_overrides(
            {
                "fallback_providers": [
                    _fallback_entry(
                        base_url=f"https://api.example.com/v1?{param}={_SK_KEY}"
                    )
                ]
            }
        )
        rendered = out["fallback_providers"]
        assert _SK_KEY not in rendered, param
        # The parameter name itself stays — operators need to see the shape of
        # the endpoint, only the value goes.
        assert "api.example.com" in rendered, param

    # A public parameter next to a secret one must survive, or the masking is
    # just a blunt instrument that costs diagnostics.
    out = dump._config_overrides(
        {
            "fallback_providers": [
                _fallback_entry(
                    base_url=f"https://api.example.com/v1?region=eu-west-1&signature={_SK_KEY}"
                )
            ]
        }
    )
    assert _SK_KEY not in out["fallback_providers"]
    assert "eu-west-1" in out["fallback_providers"]


def test_masks_credentials_carried_in_the_url_fragment():
    # The separator does not change what the value is. OAuth implicit flow
    # returns its token after "#", and the repository policy classifies
    # fragment pairs exactly like query pairs.
    for param in ("access_token", "signature", "x-amz-signature", "code", "jwt"):
        out = dump._config_overrides(
            {
                "fallback_providers": [
                    _fallback_entry(
                        base_url=f"https://api.example.com/callback#{param}={_SK_KEY}"
                    )
                ]
            }
        )
        rendered = out["fallback_providers"]
        assert _SK_KEY not in rendered, param
        assert "api.example.com" in rendered, param

    # A public fragment parameter beside a secret one must survive, otherwise
    # the assertion above would also pass on a masker that ate the fragment.
    out = dump._config_overrides(
        {
            "fallback_providers": [
                _fallback_entry(
                    base_url=(
                        f"https://api.example.com/callback#access_token={_SK_KEY}"
                        "&state=public-xyz"
                    )
                )
            ]
        }
    )
    rendered = out["fallback_providers"]
    assert _SK_KEY not in rendered
    assert "state=public-xyz" in rendered


def test_masks_userinfo_in_a_scheme_relative_endpoint():
    # `//user:pass@host/v1` parses with an authority but no scheme. Requiring a
    # scheme let that form through with the password intact.
    out = dump._config_overrides(
        {
            "fallback_providers": [
                _fallback_entry(base_url=f"//user:{_SK_KEY}@api.example.com/v1")
            ]
        }
    )
    rendered = out["fallback_providers"]
    assert _SK_KEY not in rendered
    assert "api.example.com" in rendered


def test_hyphenated_signature_names_stay_masked():
    # Guards the reason this path classifies names with
    # `is_sensitive_query_param` instead of delegating wholesale to
    # `_redact_strict_url_credentials`: that helper canonicalizes "-" to "_",
    # so `x-amz-signature` misses the (hyphenated) set entry and survives.
    from agent import redact

    assert redact.is_sensitive_query_param("x-amz-signature")
    for component in ("?", "#"):
        out = dump._config_overrides(
            {
                "fallback_providers": [
                    _fallback_entry(
                        base_url=f"https://api.example.com/o{component}x-amz-signature={_SK_KEY}"
                    )
                ]
            }
        )
        assert _SK_KEY not in out["fallback_providers"], component


def test_debug_share_upload_never_carries_the_raw_fallback_key(monkeypatch, tmp_path):
    """The real boundary: what `hermes debug share` hands to the paste service.

    The tests above pin `_config_overrides` and `run_dump`, but the credential
    only becomes public when `build_debug_share` uploads the bundle. This test
    stands at that exact call and reads the payloads that would go out.
    """
    from hermes_cli import debug
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

    uploaded: list[str] = []

    def _capture(content: str, expiry_days: int = 7) -> str:
        uploaded.append(content)
        return f"https://paste.example/{len(uploaded)}"

    monkeypatch.setattr(debug, "upload_to_pastebin", _capture)
    monkeypatch.setattr(debug, "_best_effort_sweep_expired_pastes", lambda: None)
    monkeypatch.setattr(debug, "_schedule_auto_delete", lambda urls: None)

    debug.build_debug_share(log_lines=5, redact=True)

    assert uploaded, "nothing was uploaded — the boundary was never reached"
    for payload in uploaded:
        assert _SK_KEY not in payload
    # Non-vacuous: the fallback block really is in what goes out, masked.
    assert any("sk-l...SHIP" in payload for payload in uploaded)


def test_debug_share_upload_never_carries_a_fragment_credential(monkeypatch, tmp_path):
    """The same boundary, for a credential that arrives after ``#``.

    An endpoint field is not obviously a secret field, which is what made this
    worth pinning at the upload call rather than one layer up.
    """
    from hermes_cli import debug
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
        f"    base_url: https://api.example.com/callback#access_token={_SK_KEY}&state=public-xyz\n",
        encoding="utf-8",
    )

    uploaded: list[str] = []

    def _capture(content: str, expiry_days: int = 7) -> str:
        uploaded.append(content)
        return f"https://paste.example/{len(uploaded)}"

    monkeypatch.setattr(debug, "upload_to_pastebin", _capture)
    monkeypatch.setattr(debug, "_best_effort_sweep_expired_pastes", lambda: None)
    monkeypatch.setattr(debug, "_schedule_auto_delete", lambda urls: None)

    debug.build_debug_share(log_lines=5, redact=True)

    assert uploaded, "nothing was uploaded — the boundary was never reached"
    for payload in uploaded:
        assert _SK_KEY not in payload
    # Non-vacuous on both sides: the masked token is there, and the public
    # fragment parameter next to it survived the trip.
    assert any("access_token=sk-l...SHIP" in payload for payload in uploaded)
    assert any("state=public-xyz" in payload for payload in uploaded)


def test_debug_share_upload_never_carries_a_semicolon_delimited_credential(
    monkeypatch, tmp_path
):
    """Same boundary, for a secret introduced by ``;`` instead of ``&``.

    ``parse_qsl`` splits on ``&`` only since CPython 3.9.2 (bpo-42967), so
    ``?region=eu;signature=<secret>`` used to parse as the single pair
    ``region`` — a name nobody classifies as sensitive — and the credential
    reached the paste intact.
    """
    from hermes_cli import debug
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
        f"    base_url: https://api.example.com/v1?region=eu;signature={_SK_KEY}\n",
        encoding="utf-8",
    )

    uploaded: list[str] = []

    def _capture(content: str, expiry_days: int = 7) -> str:
        uploaded.append(content)
        return f"https://paste.example/{len(uploaded)}"

    monkeypatch.setattr(debug, "upload_to_pastebin", _capture)
    monkeypatch.setattr(debug, "_best_effort_sweep_expired_pastes", lambda: None)
    monkeypatch.setattr(debug, "_schedule_auto_delete", lambda urls: None)

    debug.build_debug_share(log_lines=5, redact=True)

    assert uploaded, "nothing was uploaded — the boundary was never reached"
    for payload in uploaded:
        assert _SK_KEY not in payload
    # Non-vacuous on both sides, and the separator itself survives: a masked
    # endpoint the user cannot read back is a worse diagnostic than none.
    assert any("signature=sk-l...SHIP" in payload for payload in uploaded)
    assert any("region=eu;signature=" in payload for payload in uploaded)


def test_semicolon_delimited_fragment_credential_is_masked():
    """The ``;`` policy holds in the fragment too, not only after ``?``."""
    out = dump._config_overrides(
        {
            "fallback_providers": [
                _fallback_entry(
                    base_url=f"https://api.example.com/cb#state=public-xyz;access_token={_SK_KEY}"
                )
            ]
        }
    )

    assert _SK_KEY not in out["fallback_providers"]
    assert "access_token=sk-l...SHIP" in out["fallback_providers"]
    assert "state=public-xyz;" in out["fallback_providers"]


def test_ordinary_semicolon_parameters_are_left_alone():
    """Splitting on ``;`` must not rewrite URLs that carry no credential."""
    url = "https://api.example.com/v1?a=1;b=2&c=3"
    out = dump._config_overrides({"fallback_providers": [_fallback_entry(base_url=url)]})

    assert url in out["fallback_providers"]


def test_numeric_token_budgets_are_not_mistaken_for_secrets():
    # The name-marker rule fails closed, so the few known-safe fields that
    # contain a marker word must stay readable or the dump loses diagnostics.
    out = dump._config_overrides(
        {"fallback_providers": [_fallback_entry(max_tokens=4096, key_env="MY_KEY_VAR")]}
    )
    rendered = out["fallback_providers"]
    assert "4096" in rendered
    assert "MY_KEY_VAR" in rendered


def test_self_referencing_config_terminates():
    # A YAML anchor cycle must not turn the dump into a stack overflow.
    entry = _fallback_entry(api_key=_SK_KEY)
    entry["extra"] = entry
    out = dump._config_overrides({"fallback_providers": [entry]})
    rendered = out["fallback_providers"]
    assert _SK_KEY not in rendered

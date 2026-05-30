"""MODE B (pass:// refs -> raw value) resolution tests for
``agent.secret_sources.protonpass.fetch``, plus the argument-injection
validators and the v2/v3 hardening:

* B1 — ASCII-only env names (``^[A-Za-z_][A-Za-z0-9_]*$``); non-ASCII rejected.
* B2 — secret values keep significant leading/trailing whitespace (only the
  trailing line terminator is stripped).
* B5 — ``_split_ref`` requires the ``pass://`` scheme and EXACTLY three
  non-empty components.
* V1 — ``_is_valid_share_or_item_id`` accepts ASCII base64url WITH trailing
  ``=`` padding (real probe IDs end in ``==``).
* V2 — ``_split_ref`` rejects empty interior/leading/trailing components instead
  of collapsing them.

Split out of the former monolithic ``test_protonpass_fetch.py`` (>1000 lines);
the MODE A and C1-bootstrap/V8-B sections live in sibling modules.  Shared
fixtures/helpers come from ``tests._protonpass_helpers`` (do NOT duplicate them).
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from tests._protonpass_helpers import (  # noqa: F401
    _ok,
    _patch_run,
    _reset_caches,
    hermes_home,
    pp,
    pp_cache,
    pp_fetch,
)


# ---------------------------------------------------------------------------
# MODE B — env refs (raw stdout value)
# ---------------------------------------------------------------------------


def test_mode_b_refs_raw_value(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    captured = []

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":  # item view --field
            captured.append(cmd)
            return mock.Mock(returncode=0, stdout="sk-raw-value\n", stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"OPENAI_API_KEY": "pass://SHARE/ITEM/api_key"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-raw-value"}
    assert warnings == []
    view_cmd = captured[0]
    assert "view" in view_cmd
    assert "pass://SHARE/ITEM" in view_cmd
    assert "--field" in view_cmd
    assert view_cmd[view_cmd.index("--field") + 1] == "api_key"


def test_mode_b_nonzero_exit_with_secret_stdout_does_not_leak(
    hermes_home, monkeypatch, tmp_path
):
    """SECURITY regression: ``item view --field`` writes the bare SECRET to
    stdout, so a non-zero exit AFTER stdout was written must NOT leak it into
    the skip warning.  With empty stderr, only the exit code + a generic marker
    surface — never the captured stdout value."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    leaked = "sk-TOP-SECRET-do-not-log"

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":  # item view --field — secret on stdout, then fail
            return mock.Mock(returncode=1, stdout=leaked + "\n", stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"OPENAI_API_KEY": "pass://SHARE/ITEM/api_key"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert leaked not in warnings[0]
    assert "exited 1" in warnings[0]
    assert "OPENAI_API_KEY" in warnings[0]
    assert "(no stderr output)" in warnings[0]


def test_mode_b_partial_transient_failure_is_not_cached(
    hermes_home, monkeypatch, tmp_path
):
    """Partial-fetch caching regression: when one ref resolves but another hits a
    RECOVERABLE failure (non-zero exit / timeout), the partial result must NOT be
    cached — otherwise the failed ref is dropped for the whole TTL and never
    retried.  Expect no disk cache file and no in-process ``_CACHE`` entry."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    item_calls = {"n": 0}

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":  # first ref succeeds, second fails transiently
            item_calls["n"] += 1
            if item_calls["n"] == 1:
                return mock.Mock(returncode=0, stdout="ok-value\n", stderr="")
            return mock.Mock(returncode=1, stdout="", stderr="network error")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={
            "GOOD_KEY": "pass://SHARE/ITEM/good",
            "BAD_KEY": "pass://SHARE/ITEM/bad",
        },
        binary=binary,
        use_cache=True,            # exercise the cache-WRITE path
        cache_ttl_seconds=300,
        home_path=hermes_home,
    )

    # The good ref resolved; the bad one was skipped with a warning.
    assert secrets == {"GOOD_KEY": "ok-value"}
    assert any("BAD_KEY" in w for w in warnings)

    # The partial result must NOT be cached, so BAD_KEY retries before the TTL.
    assert not pp_cache._disk_cache_path(hermes_home).exists()
    assert pp_fetch._CACHE == {}


def test_mode_b_clean_success_is_cached(hermes_home, monkeypatch, tmp_path):
    """Over-suppression guard: when every ref resolves cleanly the result IS
    cached (the partial-fetch gate must not block normal caching)."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":
            return mock.Mock(returncode=0, stdout="v\n", stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={
            "A_KEY": "pass://SHARE/ITEM/a",
            "B_KEY": "pass://SHARE/ITEM/b",
        },
        binary=binary,
        use_cache=True,
        cache_ttl_seconds=300,
        home_path=hermes_home,
    )
    assert secrets == {"A_KEY": "v", "B_KEY": "v"}
    assert warnings == []
    assert pp_cache._disk_cache_path(hermes_home).exists()
    assert pp_fetch._CACHE  # in-process entry present


def test_mode_b_empty_value_partial_is_not_cached(hermes_home, monkeypatch, tmp_path):
    """An empty value on a success exit is treated conservatively as possibly
    transient: a partial that omits that ref must not be cached."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":
            field = cmd[cmd.index("--field") + 1]
            # GOOD resolves; EMPTY returns rc=0 with empty stdout.
            return mock.Mock(
                returncode=0,
                stdout="val\n" if field == "good" else "",
                stderr="",
            )
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={
            "GOOD_KEY": "pass://SHARE/ITEM/good",
            "EMPTY_KEY": "pass://SHARE/ITEM/empty",
        },
        binary=binary,
        use_cache=True,
        cache_ttl_seconds=300,
        home_path=hermes_home,
    )
    assert secrets == {"GOOD_KEY": "val"}
    assert any("EMPTY_KEY" in w for w in warnings)
    assert not pp_cache._disk_cache_path(hermes_home).exists()
    assert pp_fetch._CACHE == {}


def test_mode_b_failed_ref_is_retried_on_next_fetch(hermes_home, monkeypatch, tmp_path):
    """The point of not caching a partial: a ref that failed transiently is
    re-fetched on the very next call (within the TTL) and then resolves — no
    stale partial is served from cache."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    bad_calls = {"n": 0}

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":
            field = cmd[cmd.index("--field") + 1]
            if field == "good":
                return mock.Mock(returncode=0, stdout="g\n", stderr="")
            # BAD: fail transiently the first time, succeed thereafter.
            bad_calls["n"] += 1
            if bad_calls["n"] == 1:
                return mock.Mock(returncode=1, stdout="", stderr="network error")
            return mock.Mock(returncode=0, stdout="b\n", stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    refs = {"GOOD_KEY": "pass://SHARE/ITEM/good", "BAD_KEY": "pass://SHARE/ITEM/bad"}
    common = dict(
        service_token="svc", binary=binary, use_cache=True,
        cache_ttl_seconds=300, home_path=hermes_home,
    )

    first, _ = pp.fetch_protonpass_secrets(env_refs=refs, **common)
    assert first == {"GOOD_KEY": "g"}              # partial, not cached

    second, _ = pp.fetch_protonpass_secrets(env_refs=refs, **common)
    assert second == {"GOOD_KEY": "g", "BAD_KEY": "b"}  # re-fetched, now complete


def test_flag_like_vault_does_not_block_caching_of_mode_b(
    hermes_home, monkeypatch, tmp_path
):
    """A PERMANENT MODE A validation skip (a flag-like vault name, which can
    never become valid) must NOT block caching of a successful MODE B result
    fetched alongside it."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":  # MODE B view (flag-like vault is rejected pre-exec)
            return mock.Mock(returncode=0, stdout="v\n", stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        vault="--bad",   # flag-like → permanent skip with a warning
        env_refs={"GOOD_KEY": "pass://SHARE/ITEM/good"},
        binary=binary,
        use_cache=True,
        cache_ttl_seconds=300,
        home_path=hermes_home,
    )
    assert secrets == {"GOOD_KEY": "v"}
    assert any("--bad" in w for w in warnings)
    # The flag-like vault is permanent, so the good MODE B result IS cached.
    assert pp_cache._disk_cache_path(hermes_home).exists()
    assert pp_fetch._CACHE


def test_mode_a_nonzero_exit_with_secret_stdout_does_not_leak(
    hermes_home, monkeypatch, tmp_path
):
    """SECURITY regression: ``item list --show-secrets`` writes secret JSON to
    stdout, so a non-zero exit AFTER stdout was written must NOT leak that JSON
    into the MODE A skip warning, even when stderr is empty."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    leaked = json.dumps(
        {"items": [{"content": {"password": "sk-VAULT-SECRET-do-not-log"}}]}
    )

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item":  # item list --show-secrets — secret JSON, then fail
            return mock.Mock(returncode=1, stdout=leaked, stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        vault="Scoped",
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert "sk-VAULT-SECRET-do-not-log" not in warnings[0]
    assert "exited 1" in warnings[0]
    assert "Scoped" in warnings[0]
    assert "(no stderr output)" in warnings[0]


def test_mode_b_preserves_significant_whitespace(hermes_home, monkeypatch, tmp_path):
    """B2: a secret value with significant leading/trailing spaces is preserved;
    ONLY the trailing line terminator is stripped (rstrip newline, not strip)."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    # The CLI appends a newline; the value itself has meaningful padding spaces.
    raw_value = "  pad-left and trailing spaces  "

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=raw_value + "\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"PADDED": "pass://SHARE/ITEM/field"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    # The padding spaces survive; only the trailing newline is removed.
    assert secrets == {"PADDED": raw_value}
    assert warnings == []


def test_mode_b_strips_only_trailing_crlf(hermes_home, monkeypatch, tmp_path):
    """B2: a CRLF-terminated value strips only the line terminator, keeping the
    internal newline of a multi-line PEM-style value."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    pem = "-----BEGIN KEY-----\nabc\n-----END KEY-----"

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=pem + "\r\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, _ = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"PEM": "pass://SHARE/ITEM/field"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets["PEM"] == pem  # internal newlines preserved, CRLF removed


def test_mode_b_strips_only_one_trailing_newline(hermes_home, monkeypatch, tmp_path):
    """C2: a secret value that itself ends in ``\\n`` must round-trip.  pass-cli
    appends exactly ONE terminator, so the stdout is ``"secret\\n\\n"`` — the old
    ``.rstrip("\\r\\n")`` ate BOTH and corrupted the value.  Stripping EXACTLY one
    terminator keeps the value's own trailing newline."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    # The value legitimately ends in "\n"; the CLI then appends its own "\n".
    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout="secret\n\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"ENDS_NL": "pass://SHARE/ITEM/field"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {"ENDS_NL": "secret\n"}
    assert warnings == []


def test_mode_b_missing_field_skips_and_warns(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for a FIELD-less ref")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"BROKEN": "pass://SHARE/ITEM"},  # no FIELD segment
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert "BROKEN" in warnings[0]


def test_mode_b_non_pass_scheme_skips_and_warns(hermes_home, monkeypatch, tmp_path):
    """B5: a ref lacking the ``pass://`` scheme is skipped with a warning."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for a non-pass:// ref")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"K": "https://SHARE/ITEM/field"},  # wrong scheme
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert "malformed" in warnings[0]


def test_mode_b_too_many_components_skips_and_warns(hermes_home, monkeypatch, tmp_path):
    """B5: ``pass://S/I/F/extra`` is rejected (not silently truncated)."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for an over-long ref")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"K": "pass://SHARE/ITEM/FIELD/extra"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert "malformed" in warnings[0]


def test_mode_b_overrides_mode_a_on_collision(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    payload = json.dumps({
        "items": [
            {
                "content": {
                    "title": "Probe Login",
                    "content": {"Login": {"password": "from-vault"}},
                }
            }
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        if verb == "item" and "list" in cmd:
            return mock.Mock(returncode=0, stdout=payload, stderr="")
        if verb == "item" and "view" in cmd:
            return mock.Mock(returncode=0, stdout="from-ref\n", stderr="")
        return _ok()

    _patch_run(monkeypatch, fake_run)

    secrets, _ = pp.fetch_protonpass_secrets(
        service_token="svc",
        vault="My Vault",
        env_refs={"PROBE_LOGIN_PASSWORD": "pass://S/I/password"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets["PROBE_LOGIN_PASSWORD"] == "from-ref"


def test_mode_b_uses_double_dash_separator(hermes_home, monkeypatch, tmp_path):
    """A valid MODE B ref builds argv with `--` before the positional URI and
    without `--output json`."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    captured = []

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        captured.append(cmd)
        return mock.Mock(returncode=0, stdout="val\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"K": "pass://SHARE/ITEM/field"},
        binary=binary, use_cache=False, home_path=hermes_home,
    )
    cmd = captured[0]
    assert "--" in cmd
    assert "--output" not in cmd
    assert cmd.index("--") < cmd.index("pass://SHARE/ITEM")


# ---------------------------------------------------------------------------
# fetch_protonpass_secrets — guard rails
# ---------------------------------------------------------------------------


def test_fetch_empty_token_raises(hermes_home, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="service token is empty"):
        pp.fetch_protonpass_secrets(
            service_token="", vault="V", binary=binary,
            use_cache=False, home_path=hermes_home,
        )


def test_fetch_no_mode_raises(hermes_home, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="neither a vault"):
        pp.fetch_protonpass_secrets(
            service_token="svc", binary=binary,
            use_cache=False, home_path=hermes_home,
        )


# ---------------------------------------------------------------------------
# Argument-injection guard — flag-like vault / field / IDs are rejected
# ---------------------------------------------------------------------------


def test_mode_a_flag_like_vault_rejected(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("pass-cli list must not run for a flag-like vault")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc", vault="--show-secrets", binary=binary,
        use_cache=False, home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert "flag" in warnings[0]


def test_mode_b_flag_like_field_rejected(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for a flag-like FIELD")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"K": "pass://SHARE/ITEM/--evil"},
        binary=binary, use_cache=False, home_path=hermes_home,
    )
    assert secrets == {}
    assert any("flag" in w for w in warnings)


def test_mode_b_invalid_share_id_rejected(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for an invalid SHARE_ID")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"K": "pass://-bad/ITEM/field"},
        binary=binary, use_cache=False, home_path=hermes_home,
    )
    assert secrets == {}
    assert any("base64url" in w for w in warnings)


# ---------------------------------------------------------------------------
# _split_ref — strict pass:// + exactly 3 non-empty components (B5)
# ---------------------------------------------------------------------------


def test_split_ref_well_formed():
    assert pp_fetch._split_ref("pass://SHARE/ITEM/FIELD") == ("SHARE", "ITEM", "FIELD")


def test_split_ref_rejects_non_pass_scheme():
    assert pp_fetch._split_ref("https://SHARE/ITEM/FIELD") is None
    assert pp_fetch._split_ref("SHARE/ITEM/FIELD") is None
    assert pp_fetch._split_ref("") is None


def test_split_ref_rejects_wrong_component_counts():
    assert pp_fetch._split_ref("pass://SHARE/ITEM") is None        # too few
    assert pp_fetch._split_ref("pass://SHARE") is None             # too few
    assert pp_fetch._split_ref("pass://SHARE/ITEM/FIELD/extra") is None  # too many


def test_split_ref_rejects_empty_components():
    """V2: split WITHOUT filtering empty parts, then require exactly three
    NON-EMPTY components.  An empty interior or leading/trailing segment is
    REJECTED rather than "magically repaired" by collapsing the empty part."""
    # Empty interior component.
    assert pp_fetch._split_ref("pass://SHARE//FIELD") is None
    # Empty leading component.
    assert pp_fetch._split_ref("pass:///ITEM/FIELD") is None
    # Empty trailing component.
    assert pp_fetch._split_ref("pass://SHARE/ITEM/") is None
    # Multiple empty components.
    assert pp_fetch._split_ref("pass://///") is None
    # Two empties that, if filtered, would falsely "look like" 3 real parts.
    assert pp_fetch._split_ref("pass://A//") is None
    # A well-formed ref still splits cleanly.
    assert pp_fetch._split_ref("pass://A/B/C") == ("A", "B", "C")


def test_mode_b_empty_interior_component_skips_and_warns(hermes_home, monkeypatch, tmp_path):
    """V2 end-to-end: ``pass://SHARE//FIELD`` is skipped with a warning — the
    empty interior component is NOT collapsed into a 3-part ref."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for an empty-component ref")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={
            "INTERIOR": "pass://SHARE//FIELD",
            "LEADING": "pass:///ITEM/FIELD",
        },
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {}
    assert len(warnings) == 2
    assert all("malformed" in w for w in warnings)
    # N3: the warning names only the env var, never the user's actual pass://
    # ref (the warning may still carry the generic expected-shape TEMPLATE
    # ``pass://SHARE_ID/ITEM_ID/FIELD``, but not the user's concrete ref).
    assert any("INTERIOR" in w for w in warnings)
    assert any("LEADING" in w for w in warnings)
    joined = " ".join(warnings)
    assert "pass://SHARE//FIELD" not in joined
    assert "pass:///ITEM/FIELD" not in joined


# ---------------------------------------------------------------------------
# _is_valid_env_name — ASCII only (B1)
# ---------------------------------------------------------------------------


def test_is_valid_env_name_ascii_only():
    assert pp_fetch._is_valid_env_name("OPENAI_API_KEY")
    assert pp_fetch._is_valid_env_name("_X1")
    assert not pp_fetch._is_valid_env_name("1LEADING_DIGIT")
    assert not pp_fetch._is_valid_env_name("HAS-DASH")
    assert not pp_fetch._is_valid_env_name("")


def test_is_valid_env_name_rejects_non_ascii():
    """B1: a Unicode-letter "name" (which str.isalpha would accept) is rejected
    because env var names must be pure ASCII for the OS / shell."""
    assert not pp_fetch._is_valid_env_name("CAFÉ_KEY")   # E with acute accent
    assert not pp_fetch._is_valid_env_name("АБВ")  # Cyrillic letters
    assert not pp_fetch._is_valid_env_name("KEY¹")       # superscript digit


def test_mode_b_non_ascii_env_name_rejected(hermes_home, monkeypatch, tmp_path):
    """B1 end-to-end: a non-ASCII MODE B env name is skipped without invoking
    pass-cli view."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        raise AssertionError("item view must not run for a non-ASCII env name")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"CAFÉ_KEY": "pass://SHARE/ITEM/field"},
        binary=binary, use_cache=False, home_path=hermes_home,
    )
    assert secrets == {}
    assert any("valid env-var name" in w for w in warnings)


def test_is_valid_share_or_item_id():
    assert pp_fetch._is_valid_share_or_item_id("AbC_-123")
    assert pp_fetch._is_valid_share_or_item_id("_leading")
    assert not pp_fetch._is_valid_share_or_item_id("-leading-dash")
    assert not pp_fetch._is_valid_share_or_item_id("")
    assert not pp_fetch._is_valid_share_or_item_id("has/slash")
    assert not pp_fetch._is_valid_share_or_item_id("x" * (pp_fetch._MAX_ID_LEN + 1))


def test_is_valid_share_or_item_id_accepts_base64url_padding():
    """V1 (regression): real Proton IDs are base64url WITH trailing ``=``
    padding (the probe captured IDs ending in ``==``).  The v2 validator rejected
    that padding and silently skipped every real ref — these must be accepted."""
    # Real probe-shaped ID: base64url body ending in ``==``.
    assert pp_fetch._is_valid_share_or_item_id("XhBBMrgqEO90TRBZFA==")
    # A single ``=`` of padding is also valid base64url.
    assert pp_fetch._is_valid_share_or_item_id("AbCd123_-=")
    # Zero padding still fine.
    assert pp_fetch._is_valid_share_or_item_id("AbCd123_-")
    # But padding must be ONLY trailing (max two) — an embedded/leading ``=`` or
    # three ``=`` is still rejected.
    assert not pp_fetch._is_valid_share_or_item_id("AbC=def")
    assert not pp_fetch._is_valid_share_or_item_id("=AbCdef")
    assert not pp_fetch._is_valid_share_or_item_id("AbCdef===")


def test_is_valid_share_or_item_id_rejects_trailing_newline():
    """C4: a trailing ``$`` matches before a final ``\\n`` in Python, so the old
    ``^...$`` anchoring wrongly accepted ``"id\\n"``.  ``re.fullmatch`` anchors at
    end-of-string and rejects it."""
    assert not pp_fetch._is_valid_share_or_item_id("AbCd123\n")
    assert not pp_fetch._is_valid_share_or_item_id("AbCd==\n")


def test_is_valid_env_name_rejects_trailing_newline():
    """C4: the env-name validator likewise rejects a trailing newline (it now
    routes through ``re.fullmatch`` in config)."""
    assert not pp_fetch._is_valid_env_name("FOO\n")
    assert not pp_fetch._is_valid_env_name("FOO\nBAR")


def test_mode_b_resolves_real_padded_ids(hermes_home, monkeypatch, tmp_path):
    """V1 end-to-end: a ref whose SHARE/ITEM are real base64url IDs ending in
    ``==`` must resolve (was silently skipped by the v2 validator)."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    share = "XhBBMrgqEO90TRBZFA=="
    item = "9kLpQzR2sTuVwXyZ01=="
    captured = []

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        captured.append(cmd)
        return mock.Mock(returncode=0, stdout="resolved-secret\n", stderr="")

    _patch_run(monkeypatch, fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        service_token="svc",
        env_refs={"OPENROUTER_API_KEY": f"pass://{share}/{item}/api_key"},
        binary=binary,
        use_cache=False,
        home_path=hermes_home,
    )
    assert secrets == {"OPENROUTER_API_KEY": "resolved-secret"}
    assert warnings == []
    # The padded IDs survived validation and reached the positional URI.
    view_cmd = captured[0]
    assert f"pass://{share}/{item}" in view_cmd

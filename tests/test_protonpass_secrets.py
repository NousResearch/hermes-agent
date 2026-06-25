"""Hermetic tests for the Proton Pass (`pass-cli`) secret source.

We never invoke the real ``pass-cli`` binary: ``subprocess.run`` is mocked so
the suite stays fast and offline-safe.  A live resolve is exercised manually
outside of pytest.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import protonpass as pp  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_caches():
    pp._reset_cache_for_tests()
    yield
    pp._reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _clean_pass_env(monkeypatch):
    """Start every test from a known Proton Pass auth state."""
    monkeypatch.delenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", raising=False)
    yield


def _ok(value: str):
    return mock.Mock(returncode=0, stdout=value, stderr="")


def _err(code: int, stderr: str):
    return mock.Mock(returncode=code, stdout="", stderr=stderr)


def _is_login(cmd) -> bool:
    return "login" in cmd


def _ref_of(kwargs) -> str:
    """The pass:// reference a resolve invocation was asked to resolve."""
    return kwargs["env"][pp._RESOLVE_SENTINEL]


def _resolver(values):
    """A fake subprocess.run that resolves references from ``values``.

    ``values`` maps ``pass://…`` reference → stdout string.
    """

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        ref = _ref_of(kwargs)
        return _ok(values[ref])

    return fake_run


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def test_validate_references_filters_bad_names_and_refs():
    refs = {
        "OPENAI_API_KEY": "pass://Private/OpenAI/api key",
        "1BAD_NAME": "pass://Private/x/y",          # bad env name
        "HAS SPACE": "pass://Private/x/y",          # bad env name
        "NOT_A_REF": "https://example.com",          # not pass://
        "WHITESPACE": "  pass://Private/z/field  ",  # stripped + kept
    }
    valid, warnings = pp._validate_references(refs)
    assert valid == {
        "OPENAI_API_KEY": "pass://Private/OpenAI/api key",
        "WHITESPACE": "pass://Private/z/field",
    }
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# fetch_protonpass_secrets
# ---------------------------------------------------------------------------


def test_fetch_happy_path(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    values = {
        "pass://Private/OpenAI/api key": "sk-abc",
        "pass://Private/Anthropic/credential": "sk-ant-xyz",
    }
    monkeypatch.setattr(pp.subprocess, "run", _resolver(values))

    secrets, warnings = pp.fetch_protonpass_secrets(
        references={
            "OPENAI_API_KEY": "pass://Private/OpenAI/api key",
            "ANTHROPIC_API_KEY": "pass://Private/Anthropic/credential",
        },
        binary=fake,
        use_cache=False,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-abc", "ANTHROPIC_API_KEY": "sk-ant-xyz"}
    assert warnings == []


def test_fetch_uses_run_no_masking_and_option_terminator(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _ok("value")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)

    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, binary=fake, use_cache=False
    )
    cmd = captured["cmd"]
    assert cmd[:3] == [str(fake), "run", "--no-masking"]
    # `--` must precede the wrapped command so a crafted ref can't be a flag.
    assert "--" in cmd
    # The wrapped command is the current interpreter echoing the resolved
    # value — cross-platform, no POSIX-only `printenv`.
    assert cmd[cmd.index("--") + 1 :] == [sys.executable, "-c", pp._ECHO_SCRIPT]
    # The reference is passed via the child env, not on argv.
    assert captured["env"][pp._RESOLVE_SENTINEL] == "pass://V/I/F"


def test_fetch_empty_rc0_does_not_clobber(monkeypatch, tmp_path):
    """returncode 0 with empty stdout must surface as a warning, not a value."""
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(pp.subprocess, "run", lambda *a, **k: _ok("   \n"))

    secrets, warnings = pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, binary=fake, use_cache=False
    )
    assert secrets == {}
    assert any("empty value" in w for w in warnings)


def test_fetch_concealed_value_rejected(monkeypatch, tmp_path):
    """A still-masked value must never be applied as a real secret."""
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(
        pp.subprocess, "run", lambda *a, **k: _ok(pp._CONCEALED_MARKER + "\n")
    )

    secrets, warnings = pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, binary=fake, use_cache=False
    )
    assert secrets == {}
    assert any("empty value" in w for w in warnings)


def test_fetch_read_failure_becomes_warning_and_scrubs_ansi(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(
        pp.subprocess,
        "run",
        lambda *a, **k: _err(1, "\x1b[31m[ERROR] no vault access\x1b[0m"),
    )

    secrets, warnings = pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, binary=fake, use_cache=False
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert "\x1b" not in warnings[0]
    assert "[31m" not in warnings[0]
    assert "no vault access" in warnings[0]


def test_fetch_one_bad_one_good(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        ref = _ref_of(kwargs)
        return _ok("good-value") if ref == "pass://V/good/f" else _err(1, "no access")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        references={"GOOD": "pass://V/good/f", "BAD": "pass://V/bad/f"},
        binary=fake,
        use_cache=False,
    )
    assert secrets == {"GOOD": "good-value"}
    assert len(warnings) == 1


def test_fetch_auth_failure_triggers_login_and_retry(monkeypatch, tmp_path):
    """An auth-shaped failure should drive one login + a successful retry."""
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_token")
    state = {"logged_in": False, "logins": 0}

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            state["logged_in"] = True
            state["logins"] += 1
            # The token must be injected for login, never on argv.
            assert kwargs["env"].get("PROTON_PASS_PERSONAL_ACCESS_TOKEN") == "pst_token"
            assert "pst_token" not in cmd
            return _ok("")
        if not state["logged_in"]:
            return _err(1, "Error: not logged in")
        return _ok("resolved-after-login")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)

    secrets, warnings = pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, binary=fake, use_cache=False
    )
    assert secrets == {"K": "resolved-after-login"}
    assert state["logins"] == 1
    assert warnings == []


def test_fetch_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(pp, "find_pass_cli", lambda binary_path="": None)
    with pytest.raises(RuntimeError, match="pass-cli not found"):
        pp.fetch_protonpass_secrets(references={"K": "pass://V/I/F"}, use_cache=False)


def test_fetch_child_env_is_allowlisted_and_tokenless_on_resolve(monkeypatch, tmp_path):
    """The resolve child must NOT inherit provider creds or the PAT."""
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setenv("OPENAI_API_KEY", "leak-me")
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_token")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _ok("v")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, binary=fake, use_cache=False
    )
    env = captured["env"]
    assert "OPENAI_API_KEY" not in env                       # not inherited
    # The token is only for login; a resolve relies on the persistent session.
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" not in env
    assert env.get("NO_COLOR") == "1"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_inprocess_cache_hit(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._reset_cache_for_tests(tmp_path)
    for _ in range(2):
        pp.fetch_protonpass_secrets(
            references={"K": "pass://V/I/F"}, cache_ttl_seconds=60,
            binary=fake, home_path=tmp_path,
        )
    assert calls["n"] == 1  # second call served from L1 cache


def test_disk_cache_roundtrip_and_no_token_on_disk(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_supersecret")
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        calls["n"] += 1
        return _ok("resolved")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._reset_cache_for_tests(tmp_path)

    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, cache_ttl_seconds=300,
        binary=fake, home_path=tmp_path,
    )
    assert calls["n"] == 1

    cache_path = pp._disk_cache_path(tmp_path)
    assert cache_path.exists()
    assert (os.stat(cache_path).st_mode & 0o777) == 0o600
    text = cache_path.read_text()
    assert "pst_supersecret" not in text            # token never on disk
    payload = json.loads(text)
    assert payload["secrets"] == {"K": "resolved"}

    # Simulate a fresh process: clear only the in-process cache.
    pp._CACHE.clear()
    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, cache_ttl_seconds=300,
        binary=fake, home_path=tmp_path,
    )
    assert calls["n"] == 1  # served from disk, pass-cli not re-invoked


def test_ttl_zero_disables_both_layers(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._reset_cache_for_tests(tmp_path)

    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, cache_ttl_seconds=0,
        binary=fake, home_path=tmp_path,
    )
    assert not pp._disk_cache_path(tmp_path).exists()  # nothing written at TTL 0
    pp._CACHE.clear()
    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, cache_ttl_seconds=0,
        binary=fake, home_path=tmp_path,
    )
    assert calls["n"] == 2  # never cached


def test_token_change_invalidates_cache(monkeypatch, tmp_path):
    """A different personal access token must not reuse a cached value."""
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._reset_cache_for_tests(tmp_path)

    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_A")
    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, cache_ttl_seconds=300,
        binary=fake, home_path=tmp_path,
    )
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_B")
    pp._CACHE.clear()
    pp.fetch_protonpass_secrets(
        references={"K": "pass://V/I/F"}, cache_ttl_seconds=300,
        binary=fake, home_path=tmp_path,
    )
    assert calls["n"] == 2  # cache key changed → refetch


def test_partial_failure_not_cached(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")

    def fake_run(cmd, **kwargs):
        if _is_login(cmd):
            return _ok("")
        ref = _ref_of(kwargs)
        return _ok("v") if ref == "pass://V/good/f" else _err(1, "fail")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._reset_cache_for_tests(tmp_path)
    pp.fetch_protonpass_secrets(
        references={"G": "pass://V/good/f", "B": "pass://V/bad/f"},
        cache_ttl_seconds=300, binary=fake, home_path=tmp_path,
    )
    assert not pp._disk_cache_path(tmp_path).exists()


def test_reset_cache_clears_disk(tmp_path):
    cache_path = pp._disk_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{}")
    assert cache_path.exists()
    pp._reset_cache_for_tests(tmp_path)
    assert not cache_path.exists()
    pp._reset_cache_for_tests(tmp_path)  # idempotent


# ---------------------------------------------------------------------------
# find_pass_cli
# ---------------------------------------------------------------------------


def test_find_pass_cli_pinned_path_not_on_path(tmp_path, monkeypatch):
    pinned = tmp_path / "pass-cli"
    pinned.write_text("")
    pinned.chmod(0o755)
    monkeypatch.setattr(pp.shutil, "which", lambda name: "/usr/bin/pass-cli")
    assert pp.find_pass_cli(str(pinned)) == pinned


def test_find_pass_cli_pinned_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pp.shutil, "which", lambda name: "/usr/bin/pass-cli")
    assert pp.find_pass_cli(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# apply_protonpass_secrets
# ---------------------------------------------------------------------------


def test_apply_disabled_returns_empty():
    result = pp.apply_protonpass_secrets(enabled=False, env={"K": "pass://V/I/F"})
    assert result.ok
    assert not result.applied


def test_apply_missing_binary_sets_error(monkeypatch):
    monkeypatch.setattr(pp, "find_pass_cli", lambda binary_path="": None)
    result = pp.apply_protonpass_secrets(enabled=True, env={"K": "pass://V/I/F"})
    assert not result.ok
    assert "pass-cli" in result.error


def test_apply_sets_env(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(pp, "find_pass_cli", lambda binary_path="": fake)
    monkeypatch.setattr(pp.subprocess, "run", lambda *a, **k: _ok("resolved-val"))
    monkeypatch.delenv("MY_PP_KEY", raising=False)

    result = pp.apply_protonpass_secrets(
        enabled=True, env={"MY_PP_KEY": "pass://V/I/F"}, cache_ttl_seconds=0,
    )
    assert result.ok
    assert result.applied == ["MY_PP_KEY"]
    assert os.environ["MY_PP_KEY"] == "resolved-val"


def test_apply_skips_before_fetch_when_not_overriding(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(pp, "find_pass_cli", lambda binary_path="": fake)
    monkeypatch.setenv("MY_PP_KEY", "from-env")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("from-proton")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)

    result = pp.apply_protonpass_secrets(
        enabled=True, env={"MY_PP_KEY": "pass://V/I/F"},
        override_existing=False, cache_ttl_seconds=0,
    )
    assert "MY_PP_KEY" in result.skipped
    assert os.environ["MY_PP_KEY"] == "from-env"
    assert calls["n"] == 0  # never even called pass-cli for a value we'd discard


def test_apply_never_overrides_token_var(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(pp, "find_pass_cli", lambda binary_path="": fake)
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "original")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("malicious")

    monkeypatch.setattr(pp.subprocess, "run", fake_run)

    result = pp.apply_protonpass_secrets(
        enabled=True,
        env={"PROTON_PASS_PERSONAL_ACCESS_TOKEN": "pass://V/I/F"},
        override_existing=True, cache_ttl_seconds=0,
    )
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" in result.skipped
    assert os.environ["PROTON_PASS_PERSONAL_ACCESS_TOKEN"] == "original"
    assert calls["n"] == 0


def test_apply_never_raises_on_read_failure(monkeypatch, tmp_path):
    fake = tmp_path / "pass-cli"
    fake.write_text("")
    monkeypatch.setattr(pp, "find_pass_cli", lambda binary_path="": fake)
    monkeypatch.setattr(pp.subprocess, "run", lambda *a, **k: _err(1, "locked"))
    monkeypatch.delenv("MY_PP_KEY", raising=False)

    result = pp.apply_protonpass_secrets(
        enabled=True, env={"MY_PP_KEY": "pass://V/I/F"}, cache_ttl_seconds=0,
    )
    # Fail-open: warnings, nothing applied, no fatal error, no exception.
    assert result.ok
    assert result.applied == []
    assert result.warnings


def test_apply_no_valid_refs_is_noop(monkeypatch):
    # find_pass_cli must never be reached when there's nothing to fetch.
    monkeypatch.setattr(
        pp, "find_pass_cli",
        lambda binary_path="": (_ for _ in ()).throw(
            AssertionError("should not resolve pass-cli")
        ),
    )
    result = pp.apply_protonpass_secrets(enabled=True, env={"BAD NAME": "pass://V/I/F"})
    assert result.ok
    assert result.applied == []
    assert result.warnings  # the bad mapping warned

"""Tests for the github-auth skill's gh-app-token.py helper.

The helper mints short-lived (~1h) GitHub App installation tokens for git/gh
workflows. It reads App credentials from the Hermes .env FILE (never from
process env — tool subprocesses may have GITHUB_APP_* stripped), exchanges an
RS256 app JWT for an installation token, caches it with owner-only
permissions, and speaks the git credential-helper protocol.

Hermetic: in-test RSA keypair (via cryptography), httpx mocked — no network.
"""

from __future__ import annotations

import importlib.util
import io
import json
import stat
import sys
import time
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "github"
    / "github-auth"
    / "scripts"
    / "gh-app-token.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("gh_app_token_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rsa_keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return key, pem


@pytest.fixture
def app_home(tmp_path: Path, rsa_keypair) -> Path:
    """A HERMES_HOME-shaped dir: .env with App creds + the PEM key file."""
    _key, pem = rsa_keypair
    home = tmp_path / "hermes"
    home.mkdir()
    pem_path = home / "github-app.pem"
    pem_path.write_text(pem, encoding="utf-8")
    (home / ".env").write_text(
        "# hermes agent env\n"
        "GITHUB_APP_ID=123456\n"
        "GITHUB_APP_INSTALLATION_ID=987654\n"
        f"GITHUB_APP_PRIVATE_KEY_PATH={pem_path}\n",
        encoding="utf-8",
    )
    return home


def _env_file(home: Path) -> Path:
    return home / ".env"


def _cache_file(home: Path) -> Path:
    return home / "cache" / "github-app-token.json"


def _write_cache(home: Path, token: str, expires_epoch: float) -> None:
    import datetime as dt

    cache = _cache_file(home)
    cache.parent.mkdir(parents=True, exist_ok=True)
    iso = dt.datetime.fromtimestamp(expires_epoch, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    cache.write_text(json.dumps({"token": token, "expires_at": iso}), encoding="utf-8")


class _FakeResp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# .env parsing + credential reading
# ---------------------------------------------------------------------------


def test_parse_env_file_skips_comments_blanks_and_strips_quotes(mod, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nA=1\nB='two'\nC=\"three\"\nnot a kv line\nD=a=b\n",
        encoding="utf-8",
    )
    values = mod.parse_env_file(env)
    assert values == {"A": "1", "B": "two", "C": "three", "D": "a=b"}


def test_read_app_credentials_reads_required_keys(mod, app_home: Path):
    creds = mod.read_app_credentials(_env_file(app_home))
    assert creds is not None
    assert creds.app_id == "123456"
    assert creds.installation_id == "987654"
    assert creds.private_key_path == app_home / "github-app.pem"


def test_read_app_credentials_none_when_app_id_missing(mod, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_PRIVATE_KEY_PATH=/x/key.pem\n", encoding="utf-8")
    assert mod.read_app_credentials(env) is None


def test_read_app_credentials_none_when_key_path_missing(mod, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("GITHUB_APP_ID=1\nGITHUB_APP_INSTALLATION_ID=2\n", encoding="utf-8")
    assert mod.read_app_credentials(env) is None


def test_read_app_credentials_installation_id_optional(mod, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "GITHUB_APP_ID=1\nGITHUB_APP_PRIVATE_KEY_PATH=/x/key.pem\n", encoding="utf-8"
    )
    creds = mod.read_app_credentials(env)
    assert creds is not None
    assert creds.installation_id is None


def test_read_app_credentials_ignores_process_env(mod, app_home: Path, monkeypatch):
    # Tool subprocesses may have GITHUB_APP_* stripped or stale — the file wins.
    monkeypatch.setenv("GITHUB_APP_ID", "999999")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "111111")
    creds = mod.read_app_credentials(_env_file(app_home))
    assert creds.app_id == "123456"
    assert creds.installation_id == "987654"


# ---------------------------------------------------------------------------
# App JWT
# ---------------------------------------------------------------------------


def test_build_app_jwt_claims_and_alg(mod, rsa_keypair):
    import jwt as pyjwt

    key, pem = rsa_keypair
    now = 1_700_000_000
    token = mod.build_app_jwt("123456", pem, now=now)
    assert pyjwt.get_unverified_header(token)["alg"] == "RS256"
    decoded = pyjwt.decode(
        token, key.public_key(), algorithms=["RS256"], options={"verify_exp": False}
    )
    assert decoded["iss"] == "123456"
    assert decoded["iat"] == now - 60
    assert decoded["exp"] == now + 600


# ---------------------------------------------------------------------------
# Minting + installation discovery (httpx mocked)
# ---------------------------------------------------------------------------


def test_mint_posts_bearer_jwt_and_parses_201(mod, monkeypatch):
    calls = {}

    def fake_post(url, headers=None, timeout=None):
        calls["url"] = url
        calls["auth"] = headers.get("Authorization", "")
        return _FakeResp(201, {"token": "ghs_minted", "expires_at": "2099-01-01T00:00:00Z"})

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    token, expires = mod.mint_installation_token("fake.jwt.here", "987654")
    assert token == "ghs_minted"
    assert expires == "2099-01-01T00:00:00Z"
    assert calls["url"].endswith("/app/installations/987654/access_tokens")
    assert calls["auth"] == "Bearer fake.jwt.here"


def test_mint_raises_on_non_201(mod, monkeypatch):
    monkeypatch.setattr(
        mod.httpx, "post", lambda *a, **k: _FakeResp(401, {"message": "bad"})
    )
    with pytest.raises(mod.TokenError, match="401"):
        mod.mint_installation_token("fake.jwt.here", "987654")


def test_discover_single_installation(mod, monkeypatch):
    monkeypatch.setattr(
        mod.httpx, "get", lambda *a, **k: _FakeResp(200, [{"id": 555}])
    )
    assert mod.discover_installation_id("fake.jwt.here") == "555"


def test_discover_multiple_installations_errors_with_hint(mod, monkeypatch):
    payload = [
        {"id": 1, "account": {"login": "org-a"}},
        {"id": 2, "account": {"login": "org-b"}},
    ]
    monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: _FakeResp(200, payload))
    with pytest.raises(mod.TokenError, match="GITHUB_APP_INSTALLATION_ID"):
        mod.discover_installation_id("fake.jwt.here")


def test_discover_no_installations_errors(mod, monkeypatch):
    monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: _FakeResp(200, []))
    with pytest.raises(mod.TokenError, match="no installations"):
        mod.discover_installation_id("fake.jwt.here")


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_cache_hit_when_fresh(mod, app_home: Path, monkeypatch):
    _write_cache(app_home, "ghs_cached", time.time() + 3600)

    def boom(*a, **k):
        raise AssertionError("must not mint on a fresh cache hit")

    monkeypatch.setattr(mod.httpx, "post", boom)
    token = mod.get_installation_token(_env_file(app_home), _cache_file(app_home))
    assert token == "ghs_cached"


def test_remint_when_within_expiry_margin(mod, app_home: Path, monkeypatch):
    _write_cache(app_home, "ghs_stale", time.time() + 120)  # < 10 min left
    monkeypatch.setattr(
        mod.httpx,
        "post",
        lambda *a, **k: _FakeResp(201, {"token": "ghs_fresh", "expires_at": "2099-01-01T00:00:00Z"}),
    )
    token = mod.get_installation_token(_env_file(app_home), _cache_file(app_home))
    assert token == "ghs_fresh"
    # Fresh token replaces the stale cache entry
    cached = json.loads(_cache_file(app_home).read_text(encoding="utf-8"))
    assert cached["token"] == "ghs_fresh"


def test_remint_when_cache_corrupt(mod, app_home: Path, monkeypatch):
    cache = _cache_file(app_home)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        mod.httpx,
        "post",
        lambda *a, **k: _FakeResp(201, {"token": "ghs_recovered", "expires_at": "2099-01-01T00:00:00Z"}),
    )
    token = mod.get_installation_token(_env_file(app_home), _cache_file(app_home))
    assert token == "ghs_recovered"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_cache_written_owner_only(mod, app_home: Path, monkeypatch):
    monkeypatch.setattr(
        mod.httpx,
        "post",
        lambda *a, **k: _FakeResp(201, {"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}),
    )
    mod.get_installation_token(_env_file(app_home), _cache_file(app_home), force=True)
    mode = stat.S_IMODE(_cache_file(app_home).stat().st_mode)
    assert mode == 0o600


def test_cache_write_is_atomic_no_tmp_leftovers(mod, app_home: Path, monkeypatch):
    monkeypatch.setattr(
        mod.httpx,
        "post",
        lambda *a, **k: _FakeResp(201, {"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}),
    )
    mod.get_installation_token(_env_file(app_home), _cache_file(app_home), force=True)
    leftovers = [p.name for p in _cache_file(app_home).parent.iterdir() if p.name != _cache_file(app_home).name]
    assert leftovers == []


# ---------------------------------------------------------------------------
# CLI: mint
# ---------------------------------------------------------------------------


def _cli(mod, home: Path, *argv: str) -> list:
    return [
        *argv,
        "--env-file",
        str(_env_file(home)),
        "--cache-file",
        str(_cache_file(home)),
    ]


def test_mint_cli_prints_bare_token(mod, app_home: Path, monkeypatch, capsys):
    _write_cache(app_home, "ghs_cli", time.time() + 3600)
    rc = mod.main(_cli(mod, app_home, "mint"))
    out = capsys.readouterr().out
    assert rc == 0
    assert out == "ghs_cli\n"


def test_mint_cli_missing_creds_exit_1_one_stderr_line(mod, tmp_path: Path, capsys):
    home = tmp_path / "empty"
    home.mkdir()
    (home / ".env").write_text("SOME_OTHER_KEY=1\n", encoding="utf-8")
    rc = mod.main(_cli(mod, home, "mint"))
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    assert "GITHUB_APP_ID" in captured.err


# ---------------------------------------------------------------------------
# CLI: credential (git credential-helper protocol)
# ---------------------------------------------------------------------------


def _feed_stdin(monkeypatch):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("protocol=https\nhost=github.com\n\n")
    )


def test_credential_get_stdout_format(mod, app_home: Path, monkeypatch, capsys):
    _write_cache(app_home, "ghs_helper", time.time() + 3600)
    _feed_stdin(monkeypatch)
    rc = mod.main(_cli(mod, app_home, "credential", "get"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "username=x-access-token\n" in out
    assert "password=ghs_helper\n" in out
    assert out.endswith("\n\n")


def test_credential_defaults_to_get_when_no_operation(mod, app_home: Path, monkeypatch, capsys):
    _write_cache(app_home, "ghs_default", time.time() + 3600)
    _feed_stdin(monkeypatch)
    rc = mod.main(_cli(mod, app_home, "credential"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "password=ghs_default\n" in out


@pytest.mark.parametrize("operation", ["store", "erase"])
def test_credential_store_erase_silent_noop(mod, app_home: Path, monkeypatch, capsys, operation):
    def boom(*a, **k):
        raise AssertionError("store/erase must never mint")

    monkeypatch.setattr(mod.httpx, "post", boom)
    _feed_stdin(monkeypatch)
    rc = mod.main(_cli(mod, app_home, "credential", operation))
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    assert captured.err == ""


def test_credential_missing_creds_exit_1_no_stdout(mod, tmp_path: Path, monkeypatch, capsys):
    # No stdout on failure so git falls through to other configured helpers.
    home = tmp_path / "empty"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    _feed_stdin(monkeypatch)
    rc = mod.main(_cli(mod, home, "credential", "get"))
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert captured.err != ""


def _feed_stdin_for(monkeypatch, block: str):
    monkeypatch.setattr(sys, "stdin", io.StringIO(block))


@pytest.mark.parametrize(
    "block",
    [
        "protocol=https\nhost=evil.example.com\n\n",
        "protocol=https\nhost=github.com.evil.example\n\n",
        "protocol=http\nhost=github.com\n\n",  # plaintext http: never
    ],
)
def test_credential_refuses_foreign_host_or_plain_http(
    mod, app_home: Path, monkeypatch, capsys, block
):
    """Git invokes helpers for EVERY remote; the token must only ever be
    offered to GitHub over https. Foreign host -> silent exit 0 so git falls
    through, and no mint attempt is made."""

    def boom(*a, **k):
        raise AssertionError("must not mint for a foreign host")

    monkeypatch.setattr(mod.httpx, "post", boom)
    _write_cache(app_home, "ghs_leakable", time.time() + 3600)
    _feed_stdin_for(monkeypatch, block)
    rc = mod.main(_cli(mod, app_home, "credential", "get"))
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""


def test_credential_allows_gist_host(mod, app_home: Path, monkeypatch, capsys):
    _write_cache(app_home, "ghs_gist", time.time() + 3600)
    _feed_stdin_for(monkeypatch, "protocol=https\nhost=gist.github.com\n\n")
    rc = mod.main(_cli(mod, app_home, "credential", "get"))
    assert rc == 0
    assert "password=ghs_gist\n" in capsys.readouterr().out


def test_credential_no_stdin_block_still_answers(mod, app_home: Path, monkeypatch, capsys):
    """Manual invocation without a description block (not-git) still works —
    git itself always supplies host=, so absence means a human/test caller."""
    _write_cache(app_home, "ghs_manual", time.time() + 3600)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    rc = mod.main(_cli(mod, app_home, "credential", "get"))
    assert rc == 0
    assert "password=ghs_manual\n" in capsys.readouterr().out

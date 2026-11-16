"""The one-time import runs only when every kept-local ``.env`` line keeps its value; otherwise the
boot fails closed.

First thing in each profile's boot, before anything is staged, refreshed, written or sent, the
store computes each kept-local name's value now (the local file, as every reader reads it) and
afterwards (the stored entries, then the file the import leaves, a kept line frozen when that
round-trips). When any differs, the boot stops exactly as for an unreachable plane
(``SecretStoreUnavailable``): nothing sent, every file byte for byte, the error and ``hermes
doctor`` naming the profile and lines (never a value). There is no fallback to the local store.
Expected values come from running the local store on the same text.
"""
from __future__ import annotations

import datetime as dt
import json
import random

from pathlib import Path

import pytest

import hermes_secret_routing as routing
from plugins.secret_store.remote.client import bearer as _REAL_BEARER
from tests.plugins.secret_store.test_remote_load_at_start import _boot, _fresh_start, _readers

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

NAME = "GATEWAY_RELAY_IDP_CLIENT_SECRET"   # kept local by the remote store (its IdP credential)
REF = "TERMINAL_RV_REF"                    # kept local: process-global
_NAMES = ("RV_DIR", "RV_X", "RV_SHELL", NAME, REF)
REFUSED_ENV = 'RV_DIR="first\npart"\nTERMINAL_RV_REF=${RV_DIR}|\nRV_DIR=last\n'
_VALUES = ("first", "part|", "plane-only-key-7c1", "imported-key", "codex-local", "pool-local", "sk-local")


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    """The plane credential comes from the stub directly: these files reassign the IdP secret name."""
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _files(*homes):
    """Every file under *homes*, byte for byte (what "nothing written" compares)."""
    return {str(p): p.read_bytes() for home in homes for p in sorted(Path(home).rglob("*")) if p.is_file()}


def _named_without_values(err, name, *profiles):
    shown = str(err.value.code)
    assert name in shown and all(f"'{p}'" in shown for p in profiles), shown
    assert "no fallback to the local store" in shown
    for value in _VALUES:
        assert value not in shown


def _failed_closed(plane, err, name, *homes, before, profile="default"):
    """Stopped naming *name*: nothing sent, no state, no staging, every file under *homes* as before."""
    _named_without_values(err, name, profile)
    assert plane.patches() == []
    assert _files(*homes) == before
    assert not any((Path(h) / ".secrets-import.json").exists() for h in homes)


def _local_values(plane, monkeypatch, text, shell, names):
    """What the local store serves for *text*: the base every successful import must equal."""
    _fresh(monkeypatch, shell)
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    values = {name: _readers(plane.home, name) for name in names}
    monkeypatch.setenv("HERMES_SECRET_STORE", "remote")
    routing.reset_for_tests()
    return values


def _fresh(monkeypatch, shell):
    _fresh_start(monkeypatch, {})
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in shell.items():
        monkeypatch.setenv(name, value)


def _remote_boot(plane, monkeypatch, text, shell, names):
    """A remote first boot on *text*; then, as every later start, a fresh one on what it left."""
    _fresh(monkeypatch, shell)
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    first = {name: _readers(plane.home, name) for name in names}
    _fresh(monkeypatch, shell)
    _boot(plane.home)
    later = {name: _readers(plane.home, name) for name in names}
    return first, later


SPANNING = {
    "escaped_lf": 'RV_DIR="first\\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n',
    "escaped_cr": 'RV_DIR="first\\rpart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n',
    "physical_lf": 'RV_DIR="first\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n',
}


@pytest.mark.parametrize("case", SPANNING)
def test_an_import_that_would_change_a_kept_line_fails_closed_and_writes_nothing(plane, monkeypatch, case):
    """The reviewer's file: the kept line's value has a line break (it cannot be frozen) and depends on
    a name the import would collapse to its last assignment. The boot stops naming the line; the
    files, auth.json included, are byte for byte as they were; the next start stops the same way."""
    text = SPANNING[case]
    _fresh(monkeypatch, {})
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    (plane.home / "auth.json").write_text(json.dumps({"providers": {"openai-codex": {"t": "codex-local"}}}),
                                          encoding="utf-8")
    before = _files(plane.home)
    for _start in range(2):
        _fresh(monkeypatch, {})
        with pytest.raises(routing.SecretStoreUnavailable) as err:
            _boot(plane.home)
        _failed_closed(plane, err, NAME, plane.home, before=before)
        assert plane.requests == []   # not even a GET: the check needs no credential
        assert routing.routed_provider()._states == {}


def test_a_simplified_file_imports_on_the_next_boot(plane, monkeypatch):
    _fresh(monkeypatch, {})
    (plane.home / ".env").write_text(SPANNING["escaped_lf"], encoding="utf-8", newline="")
    with pytest.raises(routing.SecretStoreUnavailable):
        _boot(plane.home)
    simplified = 'RV_DIR="first\\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\n'
    first, later = _remote_boot(plane, monkeypatch, simplified, {}, [NAME])
    assert len(plane.patches()) == 1
    assert set(first[NAME].values()) == set(later[NAME].values()) == {"first\npart|"}


FROZEN_OK = {
    "spanning_one_line": ("RV_DIR=first\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\nRV_DIR=last\n", {}),
    "spanning_shell": ("GATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_SHELL}|\nRV_DIR=x\n", {"RV_SHELL": "from-shell"}),
    "multiline_single_assignment": ('RV_DIR="l1\\nl2"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\n', {}),
}


@pytest.mark.parametrize("case", FROZEN_OK)
def test_kept_lines_that_keep_their_value_still_import(plane, monkeypatch, case):
    text, shell = FROZEN_OK[case]
    base = _local_values(plane, monkeypatch, text, shell, [NAME])
    first, later = _remote_boot(plane, monkeypatch, text, shell, [NAME])
    assert len(plane.patches()) == 1
    assert first == base and later == base


def _generated():
    """``.env`` files whose kept line references an imported name, a shell name, or one assigned twice."""
    from hermes_cli.config import _quote_env_value
    rng = random.Random(0x1F0E)
    alphabet = ["a", "Z", "0", " ", "\t", "\n", "\r", '"', "'", "\\", "$", "${", "${RV_X}", "${RV_X:-d}", "#"]
    values = ["line\nsecond", "line\rsecond", "a\r\nb", "x\ty", '"quote"', "single'", "back\\slash", "$", "${",
              "${RV_X}", "#tag", "a # b", "\\n", "a"]
    values += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6))) for _ in range(20)]
    for i, value in enumerate(values):
        quoted = _quote_env_value(value)
        yield f"shell-{i}", f"{REF}=${{RV_SHELL}}|\nRV_DIR=x\n", {"RV_SHELL": value}
        yield f"file-{i}", f"RV_DIR={quoted}\n{REF}=${{RV_DIR}}|\n", {}
        yield f"spanning-{i}", f"RV_DIR={quoted}\n{REF}=${{RV_DIR}}|\nRV_DIR=last\n", {}
        yield f"spanning-raw-{i}", f'RV_DIR="{value}"\n{REF}=${{RV_DIR}}|\nRV_DIR=last\n', {}


_CASES = list(_generated())


@pytest.mark.parametrize("label,text,shell", _CASES, ids=[c[0] for c in _CASES])
def test_generated_files_either_import_with_base_values_or_fail_closed_untouched(plane, monkeypatch, label, text, shell):
    """For every generated ``.env``: the import ran and all five readers serve the local store's
    value at the import's boot and every later one, or the boot failed closed with nothing sent and
    the files byte for byte as written."""
    base = _local_values(plane, monkeypatch, text, shell, [REF])
    _fresh(monkeypatch, shell)
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    (plane.home / "auth.json").unlink(missing_ok=True)
    before = _files(plane.home)
    try:
        first, later = _remote_boot(plane, monkeypatch, text, shell, [REF])
    except routing.SecretStoreUnavailable as exc:
        assert REF in str(exc.code), label
        assert plane.patches() == [], label
        assert _files(plane.home) == before, label
        return
    assert len(plane.patches()) == 1, label
    assert len(base[REF]) == 5
    assert first == base, (label, base, first)
    assert later == base, (label, base, later)


def test_the_generated_files_include_both_outcomes(plane, monkeypatch):
    """Guard against a generator that never reaches the refusal (or never imports)."""
    from plugins.secret_store.remote import importer
    provider = routing.routed_provider()
    outcomes = set()
    for _label, text, shell in _CASES:
        _fresh(monkeypatch, shell)
        (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
        outcomes.add(bool(importer.refusal_names(plane.home, provider)))
    assert outcomes == {True, False}


# --- rv25: the check comes before any staging, refresh or write ------------------------------------

def _nous_credential(monkeypatch, refreshes):
    """The cloud credential (the Nous token in auth.json, no IdP); a refresh is counted, never sent."""
    from hermes_cli import auth
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    for name in ("GATEWAY_RELAY_IDP_TOKEN_URL", "GATEWAY_RELAY_IDP_CLIENT_ID", "GATEWAY_RELAY_IDP_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(client, "bearer", _REAL_BEARER)   # the real resolver: it would refresh

    def refresh(**_kw):
        refreshes.append(1)
        return {"access_token": TOKEN, "refresh_token": "rt-2", "expires_in": 3600}
    monkeypatch.setattr(auth, "_refresh_nous_or_quarantine", refresh)
    monkeypatch.setattr(auth, "_write_shared_nous_state", lambda *a, **k: None)
    monkeypatch.setattr(auth, "_merge_shared_nous_oauth_state", lambda state: False)
    auth._RESOLVE_TOKEN_CACHE.clear()


def _iso(seconds):
    return (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=seconds)).isoformat()


def _nous_auth(home, *, expires_in, extra=True):
    """*home*'s auth.json: the Nous record expiring in *expires_in* seconds, plus (``extra``) local
    OAuth records an import would take."""
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    nous = {"access_token": TOKEN, "refresh_token": "rt-1", "expires_at": _iso(expires_in),
            "portal_base_url": "https://portal.nousresearch.com", "client_id": "hermes-cli"}
    store = {"version": 1, "providers": {"nous": nous}}
    if extra:
        store.update(active_provider="openai-codex", credential_pool={"openrouter": [{"id": "p1", "access_token": "pool-local"}]})
        store["providers"]["openai-codex"] = {"tokens": {"access_token": "codex-local"}}
    (Path(home) / "auth.json").write_text(json.dumps(store), encoding="utf-8")


@pytest.mark.parametrize("plane_state", ["empty", "stored"])
@pytest.mark.parametrize("expires_in", [-10**6, 90, 10**6], ids=["expired-token", "refresh-window", "live-token"])
def test_a_failed_check_never_refreshes_and_keeps_every_oauth_record(plane, monkeypatch, expires_in, plane_state):
    """rv25/rv26: whatever the token's state and whether the plane already has the profile, the boot
    stops before the credential resolves: no refresh (its save would leave auth.json with the Nous
    record only), no request at all, no staging, auth.json and .env byte for byte."""
    refreshes = []
    _fresh_start(monkeypatch, {})
    _nous_credential(monkeypatch, refreshes)
    if plane_state == "stored":
        plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "plane-only-key-7c1"})
    (plane.home / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    _nous_auth(plane.home, expires_in=expires_in)
    before = _files(plane.home)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _failed_closed(plane, err, REF, plane.home, before=before)
    assert refreshes == [] and plane.requests == []


def test_a_crash_right_after_the_check_has_written_nothing(plane, monkeypatch):
    """The check runs before any staging, refresh or write: a crash right after it leaves no staging
    file and the files as they were."""
    from plugins.secret_store.remote import importer
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    _nous_auth(plane.home, expires_in=90)
    before = _files(plane.home)
    original = importer.refusal_names
    observed = []

    def crash(*args):
        observed.append(original(*args))
        raise RuntimeError("crash right after the check")
    monkeypatch.setattr(importer, "refusal_names", crash)
    with pytest.raises(RuntimeError, match="crash right after the check"):
        _boot(plane.home)
    assert observed == [[REF]]
    assert plane.patches() == [] and plane.requests == []
    assert _files(plane.home) == before


def test_hermes_doctor_names_the_lines_without_values(plane, monkeypatch, capsys):
    """``hermes doctor`` reports the boot's own stop: the profile and lines, never a value."""
    from hermes_cli import doctor, doctor_config
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text(REFUSED_ENV + "OPENAI_API_KEY=sk-local\n", encoding="utf-8", newline="")
    before = _files(plane.home)
    monkeypatch.setattr(doctor, "HERMES_HOME", plane.home)
    finding = doctor_config._check_env_file(False)
    shown = capsys.readouterr().out
    assert REF in shown and "'default'" in shown and "Secret store:" in shown
    assert any("Hermes does not start" in issue for issue in finding.issues)
    assert not any(value in shown for value in _VALUES)
    assert plane.patches() == [] and _files(plane.home) == before


# --- several profiles: each fails closed at its own import ------------------------------------------

def _sibling(plane, text, *, name="other"):
    other = plane.home.parent / name
    other.mkdir()
    (other / ".env").write_text(text, encoding="utf-8", newline="")
    return other


def test_an_unsafe_sibling_fails_closed_at_its_own_import_only(plane, monkeypatch):
    """Multiplexing: the launch profile imports cleanly and is served; the unsafe sibling is not
    looked at until its first bind, which fails closed naming it, with its files untouched and
    nothing sent. The launch profile's booted view is unaffected."""
    from agent.secret_scope import build_profile_secret_scope
    _fresh_start(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    other = _sibling(plane, REFUSED_ENV)
    sibling_before = _files(other)
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _boot(plane.home)
    assert len(plane.patches()) == 1 and _files(other) == sibling_before
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        build_profile_secret_scope(other)   # the sibling's first bind boots it
    _named_without_values(err, REF, "other")
    assert len(plane.patches()) == 1 and _files(other) == sibling_before
    assert not (other / ".secrets-import.json").exists()
    assert set(routing.routed_provider()._states) == {routing.routed_provider()._key(plane.home)}
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"imported-key"}


def test_an_imported_profile_starts_with_an_unsafe_sibling_present(plane, monkeypatch):
    """``default`` is on the plane; a sibling whose import would change a kept line appears. The
    next start of ``default`` is unaffected (no scan of other profiles); the sibling's own boot,
    and only it, fails closed."""
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _boot(plane.home)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    other = _sibling(plane, REFUSED_ENV)
    _nous_auth(other, expires_in=90)
    before = _files(other)
    _fresh_start(monkeypatch, {})
    _boot(plane.home)
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"imported-key"}
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        routing.routed_provider().boot(other)
    _named_without_values(err, REF, "other")
    assert _files(other) == before and len(plane.patches()) == 1


def test_a_hand_added_unsafe_line_after_the_import_fails_closed(plane, monkeypatch):
    """Lines added by hand after the import are a new import: the same check, the same stop."""
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _boot(plane.home)
    with open(plane.home / ".env", "a", encoding="utf-8", newline="") as f:
        f.write(REFUSED_ENV)
    before = _files(plane.home)
    _fresh_start(monkeypatch, {})
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _named_without_values(err, REF, "default")
    assert len(plane.patches()) == 1 and _files(plane.home) == before


def test_a_staging_older_than_the_plane_is_never_replayed(plane, monkeypatch):
    """A staging file left by an earlier version (or a crash) holds a value the plane has since
    replaced: the plane already has the profile, so nothing is sent and the plane's value is served;
    the stale value is only set aside for review."""
    _fresh_start(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (plane.home / ".secrets-import.json").write_text(json.dumps({"env": {"OPENAI_API_KEY": "stale"}, "auth": {}}),
                                                     encoding="utf-8")
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "current"})
    _boot(plane.home)
    assert plane.patches() == []
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"current"}
    assert not (plane.home / ".secrets-import.json").exists()
    assert "stale" in (plane.home / ".env.unimported").read_text(encoding="utf-8")


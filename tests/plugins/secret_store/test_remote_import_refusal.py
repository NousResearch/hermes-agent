"""Kept-local ``.env`` lines must be literal; a profile with one that is not fails closed.

The remote store keeps its own credential (the IdP names) and every process-global name in the
profile's ``.env``, and the dotenv loader publishes those lines as it always did. The plane stores
literal values, so a kept line may not reference another variable: at each boot of the profile,
first thing under its import lock and before anything is staged, refreshed, written or sent, a
kept line the base loader would expand (``${NAME}`` / ``${NAME:-default}``) stops the boot exactly
as an unreachable plane does (``SecretStoreUnavailable``): nothing sent, every file as the base
loader's own sanitiser leaves it, the error and ``hermes doctor`` naming the profile and lines
(never a value). There is no fallback to the local store.

With literal kept lines the import runs, and every reader serves what the local store serves for
the same file (the oracle: the local store run on the same text, in the same launch environment):
imported names are resolved at import by the loader's own rules, references among themselves, to
kept lines and to the launch environment included.
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
CID = "GATEWAY_RELAY_IDP_CLIENT_ID"
REF = "TERMINAL_RV_REF"                    # kept local: process-global
_NAMES = ("RV_DIR", "RV_A", "RV_B", "RV_C", "RV_SHELL", "RV_MARK", "TERMINAL_RV_K", "GATEWAY_RELAY_IDP_AUDIENCE",
          NAME, CID, REF)
REFUSED_ENV = "RV_DIR=first\nTERMINAL_RV_REF=${RV_DIR}|\nOPENAI_API_KEY=sk-local\n"
_VALUES = ("plane-only-key-7c1", "imported-key", "codex-local", "pool-local", "sk-local", "cid-marker", "v-marker",
           "b-marker", "shell-marker")


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    """The plane credential comes from the stub directly: these files reassign the IdP names."""
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _files(*homes):
    """Every file under *homes* but the lock files, byte for byte (what "nothing written" compares)."""
    return {str(p): p.read_bytes() for home in homes for p in sorted(Path(home).rglob("*"))
            if p.is_file() and not p.name.endswith(".lock")}


def _named_without_values(err, names, *profiles):
    shown = str(err.value.code)
    assert all(n in shown for n in names) and all(f"'{p}'" in shown for p in profiles), shown
    assert "no fallback to the local store" in shown
    for value in _VALUES:
        assert value not in shown


def _failed_closed(plane, err, names, *homes, before, profile="default"):
    """Stopped naming *names*: no request at all, no state, no staging, every file as before."""
    _named_without_values(err, names, profile)
    assert plane.requests == []
    assert _files(*homes) == before
    assert not any((Path(h) / ".secrets-import.json").exists() for h in homes)


def _fresh(monkeypatch, shell):
    _fresh_start(monkeypatch, {})
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in shell.items():
        monkeypatch.setenv(name, value)


def _write(home, text):
    """*text* as the profile ``.env``, then as the dotenv loader's sanitiser leaves it (every load,
    local store included, runs it first): the bytes a refusal must leave."""
    from hermes_cli.env_loader import _sanitize_env_file_if_needed
    (Path(home) / ".env").write_text(text, encoding="utf-8", newline="")
    _sanitize_env_file_if_needed(Path(home) / ".env")


# --- non-literal kept lines: the reviewer's cases and their kin ------------------------------------

NON_LITERAL = {
    "kept_refs_idp_forward": (f"TERMINAL_RV_K=${{{NAME}}}\n{NAME}=b-marker\nRV_MARK=m\n", ["TERMINAL_RV_K"]),
    "idp_refs_idp_forward": (f"{NAME}=${{{CID}}}x\n{CID}=cid-marker\nRV_MARK=m\n", [NAME]),
    "kept_refs_selector": ("TERMINAL_RV_K=${HERMES_CONFIG_BACKEND:-none}\nHERMES_CONFIG_BACKEND=file\nRV_MARK=m\n",
                           ["TERMINAL_RV_K"]),
    "forward_kept": (f"{NAME}=${{TERMINAL_RV_K}}\nTERMINAL_RV_K=v-marker\nRV_MARK=m\n", [NAME]),
    "backward_kept": (f"TERMINAL_RV_K=v-marker\n{NAME}=${{TERMINAL_RV_K}}\nRV_MARK=m\n", [NAME]),
    "default_only": (f"{NAME}=${{RV_MISSING:-dflt}}\nRV_MARK=m\n", [NAME]),
    "refs_imported": ("RV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR}/sub\n", [REF]),
    "refs_shell": ("TERMINAL_RV_REF=${RV_SHELL}\nRV_DIR=/x\n", [REF]),
    "quoted_ref": ('TERMINAL_RV_REF="${RV_DIR}\\tq"\nRV_DIR=/x\n', [REF]),
    "multiline_ref": ('RV_DIR="l1\\nl2"\nTERMINAL_RV_REF=${RV_DIR}|\n', [REF]),
    "two_lines": ("TERMINAL_RV_A=${RV_DIR}\nTERMINAL_RV_B=${TERMINAL_RV_A}\nRV_DIR=/x\n", ["TERMINAL_RV_A", "TERMINAL_RV_B"]),
    "nothing_to_import": (f"{CID}=cid-marker\n{NAME}=${{{CID}}}x\n", [NAME]),
}


@pytest.mark.parametrize("case", NON_LITERAL)
def test_a_non_literal_kept_line_fails_closed_and_writes_nothing(plane, monkeypatch, case):
    """The boot stops naming the lines; the files, auth.json included, are as the loader's
    sanitiser left them; no request (not even a GET); the next start stops the same way."""
    text, names = NON_LITERAL[case]
    _fresh(monkeypatch, {"RV_SHELL": "shell-marker"})
    _write(plane.home, text)
    (plane.home / "auth.json").write_text(json.dumps({"providers": {"openai-codex": {"t": "codex-local"}}}),
                                          encoding="utf-8")
    before = _files(plane.home)
    for _start in range(2):
        _fresh(monkeypatch, {"RV_SHELL": "shell-marker"})
        with pytest.raises(routing.SecretStoreUnavailable) as err:
            _boot(plane.home)
        _failed_closed(plane, err, names, plane.home, before=before)
        assert routing.routed_provider()._states == {}


def test_a_fixed_file_imports_on_the_next_boot(plane, monkeypatch):
    _fresh(monkeypatch, {})
    _write(plane.home, REFUSED_ENV)
    with pytest.raises(routing.SecretStoreUnavailable):
        _boot(plane.home)
    _write(plane.home, "RV_DIR=first\nTERMINAL_RV_REF=first|\nOPENAI_API_KEY=sk-local\n")
    _fresh(monkeypatch, {})
    _boot(plane.home)
    assert [p["body"]["set"] for p in plane.patches()] == [{"RV_DIR": "first", "OPENAI_API_KEY": "sk-local"}]
    assert (plane.home / ".env").read_text(encoding="utf-8") == "TERMINAL_RV_REF=first|\n"
    assert set(_readers(plane.home, REF).values()) == {"first|"}


@pytest.mark.parametrize("value", ["$", "${", "$NAME", "a$b", "{RV_DIR}", "$$", "end$", "${RV_DIR"])
def test_a_dollar_the_loader_does_not_expand_is_literal(plane, monkeypatch, value):
    """Literal means what the base loader leaves as written: a bare ``$NAME`` or an unclosed
    ``${`` is not a reference to it."""
    from hermes_cli.config import _quote_env_value
    _fresh(monkeypatch, {})
    _write(plane.home, f"RV_DIR=/x\nTERMINAL_RV_REF={_quote_env_value(value)}\n")
    _boot(plane.home)
    assert len(plane.patches()) == 1
    assert set(_readers(plane.home, REF).values()) == {value}


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


def _remote_boot(plane, monkeypatch, text, shell, names):
    """A remote first boot on *text*, a reload in the same process, then (as every later start) a
    fresh one on what it left."""
    _fresh(monkeypatch, shell)
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    first = {name: _readers(plane.home, name) for name in names}
    _boot(plane.home)
    reload = {name: _readers(plane.home, name) for name in names}
    _fresh(monkeypatch, shell)
    _boot(plane.home)
    later = {name: _readers(plane.home, name) for name in names}
    return first, reload, later


# --- generated differential against the local store ------------------------------------------------

KEPT = ("TERMINAL_RV_K", "GATEWAY_RELAY_IDP_AUDIENCE")
IMPORTED = ("RV_A", "RV_B", "RV_C")
_REFS = ("${RV_A}", "${RV_B:-d}", "${RV_C}", "${TERMINAL_RV_K}", "${GATEWAY_RELAY_IDP_AUDIENCE}", "${RV_SHELL}",
         "${RV_MISSING:-dflt}", "${RV_A:-}")
_PLAIN = ("a", "Z", "0", "/", "-", ".", ":", "$", "{", "}", "${", "$RV_A", "\u00e9")
_QUOTED = (*_PLAIN, " ", "#", "'", "\t", "=", "\\\\", '\\"')


def _value(rng, *, ref):
    """One raw ``.env`` value: unquoted, double- or single-quoted; with a reference when *ref*."""
    style = rng.choice(("plain", "double", "single"))
    alphabet = _PLAIN if style == "plain" else [c for c in _QUOTED if style == "double" or c not in ("'", '\\"')]
    parts = [rng.choice(alphabet) for _ in range(rng.randint(0, 4))]
    if ref:
        parts.insert(rng.randint(0, len(parts)), rng.choice(_REFS))
    body = "".join(parts)
    return {"plain": body, "double": f'"{body}"', "single": f"'{body}'"}[style]


def _generated():
    """``.env`` files of kept and imported lines in random order (duplicates, forward and backward
    references among the imported names, to kept names and to the shell); a kept line references
    something in about one file in four."""
    rng = random.Random(0x11E7)
    for i in range(80):
        lines = [f"{rng.choice(IMPORTED)}={_value(rng, ref=rng.random() < 0.6)}" for _ in range(rng.randint(1, 4))]
        for _ in range(rng.randint(1, 3)):
            lines.insert(rng.randint(0, len(lines)), f"{rng.choice(KEPT)}={_value(rng, ref=rng.random() < 0.12)}")
        yield f"gen-{i}", "".join(f"{line}\n" for line in lines), {"RV_SHELL": rng.choice(["shell", "s h", ""])}


_CASES = list(_generated())


def _base_expands_a_kept_line(text):
    """Independent of the store's check: does the base loader change any kept line's value? (It
    resolves each value's references; against an empty environment a reference always resolves to
    text shorter than itself, so equality means the value holds none.)"""
    from hermes_cli.env_loader import _resolve_dotenv_assignments, parse_dotenv_bytes
    for name, value in parse_dotenv_bytes(text.encode("utf-8")):
        if name in KEPT and value is not None:
            if _resolve_dotenv_assignments([(name, value)], {}, override=True)[name] != value:
                return True
    return False


@pytest.mark.parametrize("label,text,shell", _CASES, ids=[c[0] for c in _CASES])
def test_generated_files_import_with_the_local_values_or_fail_closed(plane, monkeypatch, label, text, shell):
    """For every generated ``.env``: with literal kept lines, the import ran once and the readers
    serve the local store's values at the import's boot, at a reload and at a later start; with a
    non-literal one, the boot failed closed with no request and the files as written."""
    names = [*KEPT, *IMPORTED]
    if _base_expands_a_kept_line(text):
        _fresh(monkeypatch, shell)
        _write(plane.home, text)
        before = _files(plane.home)
        with pytest.raises(routing.SecretStoreUnavailable) as err:
            _boot(plane.home)
        assert any(n in str(err.value.code) for n in KEPT), label
        assert plane.requests == [] and _files(plane.home) == before, label
        return
    base = _local_values(plane, monkeypatch, text, shell, names)
    # A kept name: every reader as the local store's. An imported name: the value the local loader
    # resolves (the launch load's), which the plane then serves to every reader (the local scope
    # tokenizer reads some quoted values differently from the loader; that is base behaviour).
    want = {name: base[name] if name in KEPT else dict.fromkeys(base[name], base[name]["ambient"]) for name in names}
    first, reload, later = _remote_boot(plane, monkeypatch, text, shell, names)
    assert len(plane.patches()) == 1, label
    assert first == want, (label, want, first)
    assert reload == want, (label, want, reload)
    assert later == want, (label, want, later)


def test_the_generated_files_include_both_outcomes():
    """Guard against a generator that never reaches the refusal (or never imports)."""
    outcomes = [_base_expands_a_kept_line(text) for _label, text, _shell in _CASES]
    assert 5 <= sum(outcomes) <= len(outcomes) - 40


# --- the check comes before any staging, refresh or write ------------------------------------------

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
    """Whatever the token's state and whether the plane already has the profile, the boot stops
    before the credential resolves: no refresh (its save would leave auth.json with the Nous record
    only), no request at all, no staging, auth.json and .env byte for byte."""
    refreshes = []
    _fresh(monkeypatch, {})
    _nous_credential(monkeypatch, refreshes)
    if plane_state == "stored":
        plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "plane-only-key-7c1"})
    _write(plane.home, REFUSED_ENV)
    _nous_auth(plane.home, expires_in=expires_in)
    before = _files(plane.home)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _failed_closed(plane, err, [REF], plane.home, before=before)
    assert refreshes == []


def test_a_crash_right_after_the_check_has_written_nothing(plane, monkeypatch):
    """The check runs before any staging, refresh or write: a crash right after it leaves no staging
    file and the files as they were."""
    from plugins.secret_store.remote import importer
    _fresh(monkeypatch, {})
    _write(plane.home, REFUSED_ENV)
    _nous_auth(plane.home, expires_in=90)
    before = _files(plane.home)
    original = importer.non_literal_kept_names
    observed = []

    def crash(*args):
        observed.append(original(*args))
        raise RuntimeError("crash right after the check")
    monkeypatch.setattr(importer, "non_literal_kept_names", crash)
    with pytest.raises(RuntimeError, match="crash right after the check"):
        _boot(plane.home)
    assert observed == [[REF]]
    assert plane.requests == []
    assert _files(plane.home) == before


def test_hermes_doctor_names_the_lines_without_values(plane, monkeypatch, capsys):
    """``hermes doctor`` reports the boot's own stop: the profile and lines, never a value."""
    from hermes_cli import doctor, doctor_config   # a first import reads config (and boots the store)
    _fresh(monkeypatch, {})
    plane.requests.clear()
    _write(plane.home, REFUSED_ENV)
    before = _files(plane.home)
    monkeypatch.setattr(doctor, "HERMES_HOME", plane.home)
    finding = doctor_config._check_env_file(False)
    shown = capsys.readouterr().out
    assert REF in shown and "'default'" in shown and "Secret store:" in shown
    assert any("Hermes does not start" in issue for issue in finding.issues)
    assert not any(value in shown for value in _VALUES)
    assert plane.requests == [] and _files(plane.home) == before


# --- several profiles: each fails closed at its own boot -------------------------------------------

def _sibling(plane, text, *, name="other"):
    other = plane.home.parent / name
    other.mkdir()
    _write(other, text)
    return other


def test_a_non_literal_sibling_fails_closed_at_its_own_boot_only(plane, monkeypatch):
    """Multiplexing: the launch profile imports cleanly and is served; the sibling is not looked at
    until its first bind, which fails closed naming it, with its files untouched and nothing sent.
    The launch profile's booted view is unaffected."""
    from agent.secret_scope import build_profile_secret_scope
    _fresh(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    other = _sibling(plane, REFUSED_ENV)
    sibling_before = _files(other)
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _boot(plane.home)
    assert len(plane.patches()) == 1 and _files(other) == sibling_before
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        build_profile_secret_scope(other)   # the sibling's first bind boots it
    _named_without_values(err, [REF], "other")
    assert len(plane.patches()) == 1 and _files(other) == sibling_before
    assert not (other / ".secrets-import.json").exists()
    assert set(routing.routed_provider()._states) == {routing.routed_provider()._key(plane.home)}
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"imported-key"}


def test_an_imported_profile_starts_with_a_non_literal_sibling_present(plane, monkeypatch):
    """``default`` is on the plane; a sibling with a non-literal kept line appears. The next start
    of ``default`` is unaffected (no scan of other profiles); the sibling's own boot fails closed."""
    _fresh(monkeypatch, {})
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _boot(plane.home)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    other = _sibling(plane, REFUSED_ENV)
    _nous_auth(other, expires_in=90)
    before = _files(other)
    _fresh(monkeypatch, {})
    _boot(plane.home)
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"imported-key"}
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        routing.routed_provider().boot(other)
    _named_without_values(err, [REF], "other")
    assert _files(other) == before and len(plane.patches()) == 1


def test_a_hand_added_non_literal_line_after_the_import_fails_closed(plane, monkeypatch):
    """Kept lines must stay literal after the import too: every boot checks them."""
    _fresh(monkeypatch, {})
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _boot(plane.home)
    with open(plane.home / ".env", "a", encoding="utf-8", newline="") as f:
        f.write("TERMINAL_RV_REF=${OPENAI_API_KEY}\n")
    before = _files(plane.home)
    _fresh(monkeypatch, {})
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _named_without_values(err, [REF], "default")
    assert len(plane.patches()) == 1 and _files(plane.home) == before


def test_a_staging_older_than_the_plane_is_never_replayed(plane, monkeypatch):
    """A staging file left by a crash holds a value the plane has since replaced: the plane already
    has the profile, so nothing is sent and the plane's value is served; the stale value is only set
    aside for review."""
    _fresh(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (plane.home / ".secrets-import.json").write_text(json.dumps({"env": {"OPENAI_API_KEY": "stale"}, "auth": {}}),
                                                     encoding="utf-8")
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "current"})
    _boot(plane.home)
    assert plane.patches() == []
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"current"}
    assert not (plane.home / ".secrets-import.json").exists()
    assert "stale" in (plane.home / ".env.unimported").read_text(encoding="utf-8")

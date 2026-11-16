"""The one-time import runs only when every kept-local ``.env`` line keeps its value.

First thing at store selection, before anything is staged, refreshed, written or routed, the
process computes each kept-local name's value now (the local file, as every reader reads it) and
afterwards (the stored entries, then the file the import leaves, a kept line frozen when that
round-trips). When any differs on any reader and the plane has never stored the profile, the import
is refused: while no served profile is on the plane the whole process runs on the local store,
exactly as with ``HERMES_SECRET_STORE`` unset (nothing sent, ``.env`` and ``auth.json`` byte for
byte, no poller), and ``hermes doctor`` and the boot log name the lines (never a value); once one
is, it fails closed with nothing written. Each boot checks again before it stages, refreshes or
writes. Expected values come from running the local store on the same text.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
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


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    """The plane credential comes from the stub directly: these files reassign the IdP secret name."""
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _refused_process(plane):
    """The process runs on the local store: nothing routes, the remote store holds no state, no poller."""
    assert routing.routed_provider() is None
    provider = routing.get_secret_store_provider()
    assert provider._states == {} and provider._poller is None
    assert not (plane.home / ".secrets-import.json").exists()


def _local_values(plane, monkeypatch, text, shell, names):
    """What the local store serves for *text*: the base every remote outcome must equal. The file
    the local load leaves (its sanitizer may rewrite it) is ``plane.local_bytes``."""
    _fresh(monkeypatch, shell)
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    values = {name: _readers(plane.home, name) for name in names}
    plane.local_bytes = (plane.home / ".env").read_bytes()
    monkeypatch.setenv("HERMES_SECRET_STORE", "remote")
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
def test_an_import_that_would_change_a_kept_line_is_refused_and_writes_nothing(plane, monkeypatch, caplog, case):
    """The reviewer's file: the kept line's value has a line break (it cannot be frozen) and depends on
    a name the import would collapse to its last assignment. Refused: the files are untouched and every
    reader serves the local value, on this boot and the next (which checks again)."""
    text = SPANNING[case]
    base = _local_values(plane, monkeypatch, text, {}, [NAME])
    assert set(base[NAME].values()) == {text.split('"')[1].replace("\\n", "\n").replace("\\r", "\r") + "|"}
    auth = json.dumps({"providers": {"openai-codex": {"t": "local-record"}}})
    (plane.home / "auth.json").write_text(auth, encoding="utf-8")
    caplog.set_level(logging.WARNING)
    first, later = _remote_boot(plane, monkeypatch, text, {}, [NAME])
    assert plane.patches() == []
    assert (plane.home / ".env").read_bytes() == plane.local_bytes == text.encode("utf-8")
    assert (plane.home / "auth.json").read_text(encoding="utf-8") == auth
    assert not (plane.home / ".secrets-import.json").exists()
    assert first == base and later == base
    from hermes_cli.auth import _load_auth_store
    assert _load_auth_store(plane.home / "auth.json")["providers"]["openai-codex"] == {"t": "local-record"}
    _refused_process(plane)
    refusal = routing.import_refusal()
    assert NAME in refusal and "'default'" in refusal
    assert NAME in caplog.text and "did not run" in caplog.text
    for shown in (refusal, caplog.text):
        assert "first" not in shown and "part|" not in shown and "local-record" not in shown


def test_a_refused_home_writes_to_its_local_files(plane, monkeypatch):
    from hermes_cli.config import save_env_value
    text = SPANNING["escaped_lf"]
    _remote_boot(plane, monkeypatch, text, {}, [NAME])
    save_env_value("OPENAI_API_KEY", "written-locally")
    assert plane.patches() == []
    assert "OPENAI_API_KEY=written-locally" in (plane.home / ".env").read_text(encoding="utf-8")


def test_a_simplified_file_imports_on_the_next_boot(plane, monkeypatch):
    _remote_boot(plane, monkeypatch, SPANNING["escaped_lf"], {}, [NAME])
    assert plane.patches() == []
    simplified = 'RV_DIR="first\\npart"\nGATEWAY_RELAY_IDP_CLIENT_SECRET=${RV_DIR}|\n'
    first, later = _remote_boot(plane, monkeypatch, simplified, {}, [NAME])
    assert len(plane.patches()) == 1
    assert set(first[NAME].values()) == set(later[NAME].values()) == {"first\npart|"}
    assert routing.import_refusal() is None and routing.routed_provider() is not None


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
def test_generated_files_either_import_with_base_values_or_are_refused_untouched(plane, monkeypatch, label, text, shell):
    """For every generated ``.env``: the import ran and every reader serves the local value at the
    import's boot and every later one, or it was refused with nothing sent and the file untouched
    (and the readers still serve the local value)."""
    base = _local_values(plane, monkeypatch, text, shell, [REF])
    first, later = _remote_boot(plane, monkeypatch, text, shell, [REF])
    if plane.patches():
        assert len(plane.patches()) == 1, label
    else:  # refused: the process is a local-store process (files, readers, nothing routed)
        assert (plane.home / ".env").read_bytes() == plane.local_bytes, label
        assert routing.import_refusal(), label
        _refused_process(plane)
    assert first == base, (label, base, first)
    assert later == base, (label, base, later)


def test_the_generated_files_include_both_outcomes(plane, monkeypatch):
    """Guard against a generator that never reaches the refusal (or never imports)."""
    from plugins.secret_store.remote import importer
    provider = routing.routed_provider()
    outcomes = set()
    for label, text, shell in _CASES:
        _fresh(monkeypatch, shell)
        (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
        env = {k: v for k, v in importer._dotenv_values(plane.home / ".env").items()
               if not routing.stays_local(provider, k)}
        outcomes.add(bool(importer.unmatched_kept_names(plane.home, provider, env)))
        assert bool(importer.refusal_names(plane.home, provider)) == bool(importer.unmatched_kept_names(
            plane.home, provider, env)), label
    assert outcomes == {True, False}


# --- the decision comes first, and the whole process runs local -----------------------------------

REFUSED_ENV = 'RV_DIR="first\npart"\nTERMINAL_RV_REF=${RV_DIR}|\nRV_DIR=last\n'


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


def _auth_file(plane, *, expired):
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    nous = {"access_token": TOKEN, "refresh_token": "rt-1",
            "expires_at": "2000-01-01T00:00:00+00:00" if expired else "2999-01-01T00:00:00+00:00",
            "portal_base_url": "https://portal.nousresearch.com", "client_id": "hermes-cli"}
    store = {"version": 1, "active_provider": "openai-codex",
             "providers": {"nous": nous, "openai-codex": {"tokens": {"access_token": "codex-local"}}},
             "credential_pool": {"openrouter": [{"id": "p1", "access_token": "pool-local"}]}}
    text = json.dumps(store)
    (plane.home / "auth.json").write_text(text, encoding="utf-8")
    return text


@pytest.mark.parametrize("expired", [True, False], ids=["expired-token", "live-token"])
def test_a_refused_boot_never_refreshes_and_keeps_every_oauth_record(plane, monkeypatch, expired):
    """rv25 P1 (a): the decision needs no refresh (an expired token cannot ask the plane: refused
    as is), so auth.json keeps every record byte for byte; a live token asks read-only."""
    refreshes = []
    _fresh_start(monkeypatch, {})
    _nous_credential(monkeypatch, refreshes)
    (plane.home / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    auth = _auth_file(plane, expired=expired)
    _boot(plane.home)
    assert refreshes == [] and plane.patches() == []
    assert [r["method"] for r in plane.requests] == ([] if expired else ["GET"])
    assert (plane.home / "auth.json").read_text(encoding="utf-8") == auth
    _refused_process(plane)
    from hermes_cli.auth import _load_auth_store
    loaded = _load_auth_store(plane.home / "auth.json")
    assert loaded["providers"]["openai-codex"] and loaded["credential_pool"]["openrouter"]
    assert loaded["active_provider"] == "openai-codex"


def test_a_refused_process_launch_scope_follows_its_own_writes(plane, monkeypatch):
    """rv25 P1 (b): the launch scope of a refused process is the local store's, so an own save and
    removal read back at once, exactly as with the local store."""
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
    from hermes_cli.config import remove_env_value, save_env_value
    from tui_gateway.launch_profile_policy import launch_secret_scope
    seen = {}
    for store in ("local", "remote"):
        _fresh_start(monkeypatch, {})
        monkeypatch.setenv("HERMES_SECRET_STORE", store)
        routing.reset_for_tests()
        (plane.home / ".env").write_text(REFUSED_ENV + "OPENAI_API_KEY=old-key\n", encoding="utf-8", newline="")
        _boot(plane.home)
        scope = launch_secret_scope(plane.home)
        token = set_secret_scope(scope, profile_home=str(plane.home))
        try:
            before = get_secret("OPENAI_API_KEY")
            save_env_value("OPENAI_API_KEY", "new-key")
            after = get_secret("OPENAI_API_KEY")
            remove_env_value("OPENAI_API_KEY")
            seen[store] = (type(scope).__name__, before, after, get_secret("OPENAI_API_KEY"))
        finally:
            reset_secret_scope(token)
    assert seen["remote"] == seen["local"]
    assert seen["remote"][1:] == ("old-key", "new-key", None)
    assert plane.patches() == []


@pytest.mark.parametrize("where", ["names", "decision"])
def test_a_crash_right_after_the_check_has_written_nothing(plane, monkeypatch, where):
    """rv25 P1 (c): the check runs before any staging, refresh or write; a crash right after it
    (the names computed, or the whole decision made) leaves no staging file and the files as they were."""
    from plugins.secret_store.remote import backend, importer
    _fresh_start(monkeypatch, {})
    text = 'RV_DIR="first\\npart"\nTERMINAL_RV_REF=${RV_DIR}|\nRV_DIR=last\n'
    (plane.home / ".env").write_text(text, encoding="utf-8")
    observed = []
    if where == "names":
        original = importer.refusal_names

        def crash(*args):
            observed.append(original(*args))
            raise RuntimeError("crash right after the check")
        monkeypatch.setattr(importer, "refusal_names", crash)
    else:
        original = backend.RemoteSecretStoreProvider.import_refusal

        def crash(self, homes):
            observed.append(original(self, homes))
            raise RuntimeError("crash right after the check")
        monkeypatch.setattr(backend.RemoteSecretStoreProvider, "import_refusal", crash)
    with pytest.raises(RuntimeError, match="crash right after the check"):
        _boot(plane.home)
    assert observed and REF in str(observed[0])
    assert plane.patches() == []
    assert (plane.home / ".env").read_bytes() == text.encode()
    assert not (plane.home / importer.STAGING).exists()


def test_any_refused_profile_runs_the_whole_process_local(plane, monkeypatch):
    """Multiplexing: the launch profile would import cleanly, another profile under the root is
    refused; nothing is imported for either and the launch profile is served from its file."""
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text("OPENAI_API_KEY=launch-key\n", encoding="utf-8")
    other = plane.home.parent / "other"
    other.mkdir()
    (other / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    _boot(plane.home)
    assert plane.patches() == []
    _refused_process(plane)
    assert "'other'" in routing.import_refusal() and "'default'" not in routing.import_refusal()
    assert _readers(plane.home, "OPENAI_API_KEY")["resolved"] == "launch-key"
    assert (plane.home / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=launch-key\n"


def test_a_plane_that_already_has_the_profile_is_not_a_refusal(plane, monkeypatch):
    """The import would not run (profileVersion != 0): the plane wins as always, nothing local."""
    _fresh_start(monkeypatch, {})
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": "from-plane"})
    (plane.home / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    _boot(plane.home)
    assert routing.import_refusal() is None and routing.routed_provider() is not None
    assert (plane.home / ".env.unimported").exists()


def test_an_import_that_still_would_change_a_kept_line_at_boot_fails_closed(plane, monkeypatch):
    """The plane emptied between the check and the boot (or a profile the check never saw): the
    boot does not import; it stops, naming the line, and the next start's check runs it local."""
    from plugins.secret_store.remote import backend
    _fresh_start(monkeypatch, {})
    monkeypatch.setattr(backend.RemoteSecretStoreProvider, "import_refusal", lambda self, homes: None)
    (plane.home / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    with pytest.raises(routing.SecretStoreUnavailable, match=REF):
        _boot(plane.home)
    assert plane.patches() == []
    assert (plane.home / ".env").read_text(encoding="utf-8") == REFUSED_ENV
    assert not (plane.home / ".secrets-import.json").exists()


def test_hermes_doctor_reports_the_refusal_and_checks_the_local_files(plane, monkeypatch, capsys):
    """``hermes doctor`` (its own process: it decides again) names the lines, never a value, and
    goes on with the local-store checks of the files the process runs on."""
    from hermes_cli import doctor, doctor_config
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text(REFUSED_ENV + "OPENAI_API_KEY=sk-local\n", encoding="utf-8", newline="")
    monkeypatch.setattr(doctor, "HERMES_HOME", plane.home)
    finding = doctor_config._check_env_file(False)
    shown = capsys.readouterr().out
    assert REF in shown and ".env file exists" in shown and "Secret store:" not in shown
    assert any("secrets plane import" in issue for issue in finding.issues)
    assert "first" not in shown and "sk-local" not in shown
    assert plane.patches() == [] and not (plane.home / ".secrets-import.json").exists()


# --- rv26: never local once a served profile is on the plane; the check runs before any write -------

PLANE_KEY = "plane-only-key-7c1"


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
    text = json.dumps(store)
    (Path(home) / "auth.json").write_text(text, encoding="utf-8")
    return text


def _sibling(plane, text, *, name="other"):
    other = plane.home.parent / name
    other.mkdir()
    (other / ".env").write_text(text, encoding="utf-8", newline="")
    return other


def _files(*homes):
    """Every file under *homes*, byte for byte (what "nothing written" compares)."""
    return {str(p): p.read_bytes() for home in homes for p in sorted(Path(home).rglob("*")) if p.is_file()}


def _named_without_values(err, *profiles):
    shown = str(err.value.code)
    assert REF in shown and all(f"'{p}'" in shown for p in profiles), shown
    for value in ("first", "part|", PLANE_KEY, "imported-key", "codex-local", "pool-local"):
        assert value not in shown


@pytest.mark.parametrize("sibling", ["tokenless", "live-token"])
def test_a_refused_sibling_fails_closed_when_the_launch_profile_is_on_the_plane(plane, monkeypatch, sibling):
    """rv26 P1: the launch profile is already on the plane (its files hold only the bootstrap), and
    another profile's import would change a kept line. Running local would serve the launch profile
    none of its secrets, so store selection fails closed naming the line; nothing is written."""
    refreshes = []
    _fresh_start(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _nous_credential(monkeypatch, refreshes)
    plane.set_level(plane.profile("default"), env={"OPENAI_API_KEY": PLANE_KEY})
    (plane.home / ".env").write_text("TERMINAL_ENV=local\n", encoding="utf-8")
    _nous_auth(plane.home, expires_in=10**6, extra=False)
    other = _sibling(plane, REFUSED_ENV)
    if sibling == "live-token":
        _nous_auth(other, expires_in=10**6)
    before = _files(plane.home.parent)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _named_without_values(err, "other", "default")
    assert _files(plane.home.parent) == before
    assert plane.patches() == [] and refreshes == []
    assert {r["method"] for r in plane.requests} == {"GET"}
    assert __import__("os").environ.get("OPENAI_API_KEY") is None


def _import_default(plane, monkeypatch, *, expires_in=-10**6):
    """An actual one-time import of ``default`` (plane credential from the stub): its files are
    left with the bootstrap only."""
    _fresh_start(monkeypatch, {})
    (plane.home / ".env").write_text("OPENAI_API_KEY=imported-key\n", encoding="utf-8")
    _nous_auth(plane.home, expires_in=expires_in, extra=False)
    _boot(plane.home)
    assert routing.routed_provider() is not None and len(plane.patches()) == 1
    assert "imported-key" not in (plane.home / ".env").read_text(encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.mark.parametrize("evidence", ["plane", "marker-only", "staging"])
def test_an_imported_profile_with_a_later_refused_sibling_fails_closed(plane, monkeypatch, evidence):
    """rv26 P1: ``default`` was imported, then a sibling whose import would change a kept line
    appears. The next start fails closed instead of running local (which would serve ``default``
    nothing). With no unexpired token anywhere the plane cannot be asked: the local import marker
    decides. A leftover staging (an import that stopped partway) counts as imported too."""
    refreshes = []
    if evidence == "staging":
        _fresh_start(monkeypatch, {})
        (plane.home / ".secrets-import.json").write_text(json.dumps({"env": {"OPENAI_API_KEY": "staged"}, "auth": {}}),
                                                         encoding="utf-8")
    else:
        _import_default(plane, monkeypatch)
    _sibling(plane, REFUSED_ENV)
    _fresh_start(monkeypatch, {})
    if evidence == "marker-only":
        _nous_credential(monkeypatch, refreshes)   # default's token expired, the sibling has none
    plane.requests.clear()
    before = _files(plane.home.parent)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _named_without_values(err, "other", "default")
    assert _files(plane.home.parent) == before
    assert plane.patches() == [] and refreshes == []
    methods = {r["method"] for r in plane.requests}
    assert methods == set() if evidence == "marker-only" else methods == {"GET"}


def test_an_imported_profile_with_a_hand_added_line_and_an_expired_token_serves_the_plane(plane, monkeypatch):
    """The import marker says ``default`` is on the plane, so its own refusal-shaped lines (added by
    hand afterwards) never run it local even when the token cannot ask the plane: it refreshes once,
    the plane wins and serves its value, nothing is sent."""
    refreshes = []
    _import_default(plane, monkeypatch)
    with open(plane.home / ".env", "a", encoding="utf-8", newline="") as f:
        f.write(REFUSED_ENV)
    _fresh_start(monkeypatch, {})
    _nous_credential(monkeypatch, refreshes)
    _boot(plane.home)
    assert routing.import_refusal() is None and routing.routed_provider() is not None
    assert len(refreshes) == 1 and len(plane.patches()) == 1
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {"imported-key"}


def _empty_the_plane_after_selection(plane, monkeypatch):
    """Store selection sees the stored profile; the plane is emptied right after it."""
    from plugins.secret_store.remote import backend
    from tests.plugins.secret_store.stub_secrets_plane import INSTANCE
    original = backend.RemoteSecretStoreProvider.import_refusal
    seen = []

    def then_empty(self, homes):
        seen.append(original(self, homes))
        plane.profiles[(INSTANCE, "default")] = {"env": {}, "auth": {}, "version": 0}
        return seen[-1]
    monkeypatch.setattr(backend.RemoteSecretStoreProvider, "import_refusal", then_empty)
    return seen


def _credential(monkeypatch, mode, refreshes):
    if mode != "idp":
        _nous_credential(monkeypatch, refreshes)


_EXPIRES = {"idp": 10**6, "nous-live": 10**6, "nous-window": 90}


@pytest.mark.parametrize("mode", ["idp", "nous-live", "nous-window"])
def test_a_plane_emptied_after_selection_fails_closed_before_anything_is_written(plane, monkeypatch, mode):
    """rv26 P1: the check runs again before any staging, refresh or write. A Nous token inside its
    refresh window is not refreshed (that save would leave auth.json with the Nous record only), no
    staging file is left, and the files are byte for byte as they were."""
    refreshes = []
    _fresh_start(monkeypatch, {})
    _credential(monkeypatch, mode, refreshes)
    plane.set_level(plane.profile("default"), env={"PLANE_ONLY": "plane-val"})
    (plane.home / ".env").write_text(REFUSED_ENV, encoding="utf-8", newline="")
    _nous_auth(plane.home, expires_in=_EXPIRES[mode])
    seen = _empty_the_plane_after_selection(plane, monkeypatch)
    before = _files(plane.home)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    assert seen == [None]
    _named_without_values(err, "default")
    assert _files(plane.home) == before
    assert plane.patches() == [] and refreshes == []


@pytest.mark.parametrize("mode", ["idp", "nous-window"])
def test_after_a_plane_emptied_at_boot_the_next_start_equals_the_local_store(plane, monkeypatch, mode):
    from hermes_cli import auth as auth_mod
    names = ("TERMINAL_RV_REF", "RV_DIR", "OPENAI_API_KEY")
    refreshes = []
    _fresh_start(monkeypatch, {})
    _credential(monkeypatch, mode, refreshes)
    plane.set_level(plane.profile("default"), env={"PLANE_ONLY": "plane-val"})
    text = REFUSED_ENV + "OPENAI_API_KEY=old-key\n"
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    auth = _nous_auth(plane.home, expires_in=_EXPIRES[mode])
    _empty_the_plane_after_selection(plane, monkeypatch)
    with pytest.raises(routing.SecretStoreUnavailable):
        _boot(plane.home)
    _fresh_start(monkeypatch, {})
    _credential(monkeypatch, mode, refreshes)
    _boot(plane.home)
    assert routing.routed_provider() is None and routing.import_refusal()
    remote = ({n: _readers(plane.home, n) for n in names}, sorted(auth_mod._load_auth_store().get("providers", {})))
    _fresh_start(monkeypatch, {})
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    (plane.home / "auth.json").write_text(auth, encoding="utf-8")
    _boot(plane.home)
    local = ({n: _readers(plane.home, n) for n in names}, sorted(auth_mod._load_auth_store().get("providers", {})))
    assert remote == local and refreshes == []


def test_a_secret_deleted_while_local_is_never_imported_from_an_old_staging(plane, monkeypatch):
    """rv26 P1: no staging survives a failed check, so a secret removed while the process ran
    local is not sent when a later start imports."""
    from hermes_cli.config import remove_env_value
    _fresh_start(monkeypatch, {})
    plane.set_level(plane.profile("default"), env={"PLANE_ONLY": "plane-val"})
    (plane.home / ".env").write_text(REFUSED_ENV + "OPENAI_API_KEY=old-key\n", encoding="utf-8", newline="")
    _empty_the_plane_after_selection(plane, monkeypatch)
    with pytest.raises(routing.SecretStoreUnavailable):
        _boot(plane.home)
    assert not (plane.home / ".secrets-import.json").exists()
    _fresh_start(monkeypatch, {})
    _boot(plane.home)
    assert routing.routed_provider() is None
    remove_env_value("OPENAI_API_KEY")
    (plane.home / ".env").write_text("RV_DIR=last\nTERMINAL_RV_REF=plain|\n", encoding="utf-8")
    _fresh_start(monkeypatch, {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _boot(plane.home)
    assert routing.routed_provider() is not None and len(plane.patches()) == 1
    assert "OPENAI_API_KEY" not in json.dumps(plane.patches()[0]["body"])
    assert set(_readers(plane.home, "OPENAI_API_KEY").values()) == {None}


def test_a_profile_created_after_selection_fails_closed_before_its_token_refreshes(plane, monkeypatch):
    refreshes = []
    _fresh_start(monkeypatch, {})
    _nous_credential(monkeypatch, refreshes)
    (plane.home / ".env").write_text("OPENAI_API_KEY=k\n", encoding="utf-8")
    plane.set_level(plane.profile("default"), env={"PLANE_ONLY": "v"})
    _nous_auth(plane.home, expires_in=10**6)
    _boot(plane.home)
    provider = routing.routed_provider()
    assert provider is not None
    later = _sibling(plane, REFUSED_ENV, name="later")
    _nous_auth(later, expires_in=90)
    before = _files(later)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        provider.boot(later)
    _named_without_values(err, "later")
    assert _files(later) == before and refreshes == []
    assert plane.patches() == []


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

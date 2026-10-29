"""How the exact-value registry loads: concurrent first readers, bad entries, frozen workers."""

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import agent.redact_known_values as known
from agent.redact import redact_sensitive_text
from agent.redact_known_values import MARKER, clear_known_secret_values, redact_known_secret_values

# Built by concatenation: opaque, no vendor prefix, never a real credential.
ENV_VAL = "opaqueEnv" + "Value8812qz"
SNAPSHOT_VAL = "frozenSnap" + "shotKey4471x"


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_known_secret_values()
    yield
    clear_known_secret_values()


def test_concurrent_first_reader_waits_for_the_load_in_progress(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow_file_values(home):
        entered.set()
        release.wait(10)
        return [ENV_VAL]

    monkeypatch.setattr(known, "_file_values", slow_file_values)
    text = f"server echoed {ENV_VAL}"
    results = {}
    first = threading.Thread(target=lambda: results.update(first=redact_known_secret_values(text, home=tmp_path)))
    first.start()
    assert entered.wait(10)
    second = threading.Thread(target=lambda: results.update(second=redact_known_secret_values(text, home=tmp_path)))
    second.start()
    second.join(0.3)  # a reader that does not wait returns the empty pre-load set here
    release.set()
    first.join(10)
    second.join(10)
    assert results == {"first": f"server echoed {MARKER}", "second": f"server echoed {MARKER}"}


def test_malformed_mcp_url_does_not_stop_other_sources(tmp_path):
    (tmp_path / ".env").write_text(f"ACME_SERVICE_TOKEN={ENV_VAL}\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "mcp_servers:\n  broken:\n    enabled: false\n    url: 'https://[broken/?k=x'\n", encoding="utf-8")
    assert redact_known_secret_values(f"value {ENV_VAL}", home=tmp_path) == f"value {MARKER}"


def test_bracketed_authorization_value_is_masked_but_marker_is_kept():
    opaque = "opaqueCredential" + "12345abcde"
    out = redact_sensitive_text(f"Authorization: [{opaque}]", force=True)
    assert opaque not in out
    assert redact_sensitive_text("Authorization: [REDACTED]", force=True) == "Authorization: [REDACTED]"


_BINDS = {
    "safe_mode": "_bind_safe_worker_policy(safe_mode=True, ignore_user_config=True, config=CONFIG)",
    "ignore_user_config": "_bind_safe_worker_policy(safe_mode=False, ignore_user_config=True, config=CONFIG)",
    "ordinary": "_bind_worker_config(CONFIG)",
}


@pytest.mark.parametrize("bind", sorted(_BINDS))
def test_frozen_worker_masks_its_snapshot_without_opening_profile_files(tmp_path, bind):
    # The owner passes the profile's .env credentials in the worker's environment
    # (session_managed_worker._worker_env); the worker masks them without opening .env.
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"ACME_SERVICE_TOKEN={ENV_VAL}\n", encoding="utf-8")
    (home / "config.yaml").write_text("model:\n  api_key: fileOnlyValue99231x\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    script = home / "probe.py"
    script.write_text(f"import sys; sys.path.insert(0, {str(root)!r})\n" + textwrap.dedent(f"""
        import json, sys
        from agent.safe_worker_policy import _bind_safe_worker_policy, _bind_worker_config
        CONFIG = {{"model": {{"api_key": {SNAPSHOT_VAL!r}}}}}
        {_BINDS[bind]}
        opens = []
        def audit(event, args):
            if event == "open" and str(args[0]).endswith((".env", "config.yaml")):
                opens.append(str(args[0]))
        sys.addaudithook(audit)
        from agent.redact import redact_sensitive_text
        out = redact_sensitive_text("echo {SNAPSHOT_VAL} {ENV_VAL}", force=True)
        print(json.dumps({{"out": out, "opens": opens}}))
    """), encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home), ACME_SERVICE_TOKEN=ENV_VAL)
    result = subprocess.run([sys.executable, str(script)], env=env, cwd=root, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe == {"out": f"echo {MARKER} {MARKER}", "opens": []}


def test_a_live_connections_values_survive_eviction_until_its_pin_is_dropped(tmp_path):
    live = "liveConnection" + "Cred5521qq"
    pin = known.register_secret_values([live], home=tmp_path)
    newer = [f"rotatedValue{i:06d}xx" for i in range(known._RUNTIME_MAX_PER_PROFILE + 5)]
    known.register_secret_values(newer, home=tmp_path)
    assert redact_known_secret_values(f"echo {live}", home=tmp_path) == f"echo {MARKER}"
    del pin  # the connection ended: the value ages out like any other
    known.register_secret_values(["afterRelease" + "Value9911zz"], home=tmp_path)
    assert redact_known_secret_values(f"echo {live}", home=tmp_path) == f"echo {live}"

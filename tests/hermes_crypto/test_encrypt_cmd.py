"""CLI handler tests for ``hermes encrypt rotate-key --full`` confirmation."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from hermes_crypto import keystore, migrate

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


def _seed_and_enable(*, with_recovery: bool = False) -> None:
    home = get_hermes_home()
    (home / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}),
        encoding="utf-8",
    )
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    if with_recovery:
        keystore.add_recovery_slot()


def _full_rotate_args(**overrides) -> Namespace:
    base = {
        "full": True,
        "yes": False,
        "force": True,
        "keep_backups": False,
        "key_source": None,
    }
    base.update(overrides)
    return Namespace(**base)


def test_rotate_key_full_aborts_without_yes(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    called = {"full_rekey": False}

    def _fail_full_rekey(*_args, **_kwargs):
        called["full_rekey"] = True
        raise AssertionError("full_rekey should not run without confirmation")

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _fail_full_rekey)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_k: "pw")

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args())

    assert rc == 0
    assert called["full_rekey"] is False


def test_rotate_key_full_refuses_non_interactive_without_yes(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    called = {"full_rekey": False}

    def _fail_full_rekey(*_args, **_kwargs):
        called["full_rekey"] = True
        raise AssertionError("full_rekey should not run without --yes")

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _fail_full_rekey)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: False)

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args())

    assert rc == 1
    assert called["full_rekey"] is False


def test_rotate_key_full_proceeds_with_yes(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable(with_recovery=True)
    called = {"full_rekey": False}

    def _stub_full_rekey(*_args, **_kwargs):
        called["full_rekey"] = True
        return SimpleNamespace(
            rekeyed_files=[".env"],
            rekeyed_databases=[],
            rekeyed_sessions=[],
            rekeyed_logs=[],
            recovery_slots_dropped=1,
            skipped=[],
            backups_removed=0,
        )

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _stub_full_rekey)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_k: "pw")

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args(yes=True))

    assert rc == 0
    assert called["full_rekey"] is True


def test_rotate_key_full_prompts_then_proceeds_on_yes(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    called = {"full_rekey": False}

    def _stub_full_rekey(*_args, **_kwargs):
        called["full_rekey"] = True
        return SimpleNamespace(
            rekeyed_files=[".env"],
            rekeyed_databases=[],
            rekeyed_sessions=[],
            rekeyed_logs=[],
            recovery_slots_dropped=0,
            skipped=[],
            backups_removed=0,
        )

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _stub_full_rekey)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_k: "pw")

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args())

    assert rc == 0
    assert called["full_rekey"] is True


# ---------------------------------------------------------------------------
# --key-source validation must run BEFORE full_rekey().
# A typoed new passphrase (or missing keyring backend) must abort the rekey
# while the old DEK still owns every encrypted artifact on disk — otherwise
# the DEK gets rotated and the operator is locked into a half-finished
# key-source switch.
# ---------------------------------------------------------------------------


def _seed_and_enable_keyring() -> None:
    """Seed an encrypted home with key_source=keyring (for new-passphrase tests)."""
    home = get_hermes_home()
    (home / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}, "credential_pool": {}}),
        encoding="utf-8",
    )
    migrate.enable("keyring", argon2_params=FAST_ARGON2, force=True)


def test_rotate_key_full_key_source_aborts_before_full_rekey_on_bad_new_passphrase(
    monkeypatch,
):
    """A mismatched new passphrase must abort BEFORE full_rekey() runs.

    Setup: keystore is in keyring mode; operator runs ``rotate-key --full
    --key-source passphrase`` and typos the confirmation. The pre-validation
    helper must catch this and short-circuit before full_rekey()
    rotates the DEK on disk.
    """
    from hermes_cli import encrypt_cmd

    _seed_and_enable_keyring()
    keystore.unlock()
    old_dek = keystore.get_cached_dek()
    assert old_dek is not None
    # Re-lock so cmd_rotate_key follows its normal unlock path (keyring mode
    # does not prompt the user, so no getpass call for the current source).
    keystore.lock()

    called = {"full_rekey": 0}

    def _spy_full_rekey(*_args, **_kwargs):
        called["full_rekey"] += 1
        raise AssertionError(
            "full_rekey must not run when the new passphrase failed validation"
        )

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _spy_full_rekey)
    # In keyring mode the unlock path does not call getpass, so the two
    # getpass calls below correspond to _prompt_new_passphrase: first the
    # "new" prompt, then the (mismatched) confirmation.
    answers = iter(["newpw1", "newpw2_mismatch"])
    monkeypatch.setattr(
        encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: next(answers)
    )

    rc = encrypt_cmd.cmd_rotate_key(
        _full_rotate_args(yes=True, key_source="passphrase")
    )

    assert rc == 1, "must exit 1 on mismatched new passphrase"
    assert called["full_rekey"] == 0, (
        "full_rekey() must not run when new passphrase validation failed"
    )
    # Old DEK and slot type unchanged on disk.
    keystore.unlock()
    assert keystore.get_cached_dek() == old_dek
    assert keystore.primary_slot_type() == "keyring"


def test_rotate_key_full_key_source_succeeds_with_valid_new_passphrase(monkeypatch):
    """Matching new passphrase completes full rekey AND key-source switch.

    Starts in keyring mode and rotates to passphrase, exercising the new
    upfront-collection path: _prepare_new_key_source captures the new
    passphrase, full_rekey runs (stubbed), then _finish_key_source_rotation
    re-wraps the DEK under the pre-collected passphrase.
    """
    from hermes_cli import encrypt_cmd

    _seed_and_enable_keyring()

    rekey_called = {"n": 0}

    def _stub_full_rekey(*_args, **_kwargs):
        rekey_called["n"] += 1
        # The stub does not need to install a fresh DEK — the assertions
        # below only care that the slot type flipped to the new source.
        return SimpleNamespace(
            rekeyed_files=[".env"],
            rekeyed_databases=[],
            rekeyed_sessions=[],
            rekeyed_logs=[],
            recovery_slots_dropped=0,
            skipped=[],
            backups_removed=0,
        )

    # _finish_key_source_rotation calls migrate._set_config, which writes the
    # YAML config. Stub it so the test does not depend on a config file
    # existing under the isolated HERMES_HOME.
    set_config_calls = []

    def _stub_set_config(path, value):
        set_config_calls.append((path, value))

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _stub_full_rekey)
    monkeypatch.setattr("hermes_crypto.migrate._set_config", _stub_set_config)
    # Use the fast-Argon2 profile so the rotate_primary call inside
    # _finish_key_source_rotation doesn't run the production-cost KDF.
    monkeypatch.setattr(encrypt_cmd, "_argon2_params", lambda: FAST_ARGON2)
    # Keyring mode: no current-passphrase prompt. _prompt_new_passphrase
    # asks twice (matched).
    answers = iter(["brand-new-pw", "brand-new-pw"])
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: next(answers)
    )

    rc = encrypt_cmd.cmd_rotate_key(
        _full_rotate_args(yes=True, key_source="passphrase")
    )

    assert rc == 0
    assert rekey_called["n"] == 1
    assert set_config_calls == [("security.encryption.key_source", "passphrase")]
    # Slot type flipped to the new source; the new passphrase can unlock it.
    assert keystore.primary_slot_type() == "passphrase"
    keystore.lock()
    keystore.unlock(passphrase="brand-new-pw")
    assert keystore.get_cached_dek() is not None


def test_rotate_key_full_key_source_keyring_aborts_when_backend_insecure(
    monkeypatch,
):
    """Missing OS-keyring backend must abort BEFORE full_rekey()."""
    from hermes_cli import encrypt_cmd

    _seed_and_enable()

    called = {"full_rekey": 0}

    def _spy_full_rekey(*_args, **_kwargs):
        called["full_rekey"] += 1
        raise AssertionError(
            "full_rekey must not run when the keyring backend is missing"
        )

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _spy_full_rekey)
    monkeypatch.setattr("hermes_crypto.keystore.keyring_is_secure", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: "pw")

    rc = encrypt_cmd.cmd_rotate_key(
        _full_rotate_args(yes=True, key_source="keyring")
    )

    assert rc == 1
    assert called["full_rekey"] == 0


# ---------------------------------------------------------------------------
# end-to-end ``--full --key-source`` switch WITHOUT stubbing full_rekey.
# The sibling tests above all stub the rekey to keep tests fast; this one
# exercises the real envelope/keystore flow so a regression in either
# full_rekey() or its CLI wiring is caught.
# ---------------------------------------------------------------------------


def test_rotate_key_full_key_source_end_to_end_keyring_to_passphrase(
    monkeypatch, capsys,
):
    """Real full re-key + key-source switch (keyring -> passphrase).

    No stubbing of ``migrate.full_rekey`` — exercises the actual envelope
    decrypt/re-encrypt round trip plus the real ``_finish_key_source_rotation``
    re-wrap. Verifies that after the CLI returns:

    * exit code 0
    * the primary slot is now of type ``passphrase`` and unlocks under the
      new passphrase
    * the previously-seeded ``.env`` decrypts under the freshly-cached DEK
      and yields the original cleartext
    * ``backups_removed > 0`` — proves the rekey-run backup cleanup ran,
      which is what new CLI output line surfaces
    """
    from pathlib import Path

    from hermes_cli import encrypt_cmd
    from hermes_crypto import decrypt_if_encrypted

    _seed_and_enable_keyring()
    env_path = get_hermes_home() / ".env"
    assert env_path.is_file()

    # ``_finish_key_source_rotation`` calls ``migrate._set_config`` to persist
    # the new key_source to YAML. The isolated HERMES_HOME has no config file,
    # so stub it the same way the sibling stubbed test does.
    set_config_calls: list = []
    monkeypatch.setattr(
        "hermes_crypto.migrate._set_config",
        lambda path, value: set_config_calls.append((path, value)),
    )
    # Keep Argon2 cheap for the rotate_primary call inside the finisher.
    monkeypatch.setattr(encrypt_cmd, "_argon2_params", lambda: FAST_ARGON2)
    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)

    # Keyring mode means no current-passphrase prompt; getpass is only invoked
    # twice for the new passphrase (matched).
    answers = iter(["e2e-new-pw", "e2e-new-pw"])
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: next(answers)
    )

    rc = encrypt_cmd.cmd_rotate_key(
        _full_rotate_args(yes=True, key_source="passphrase")
    )

    assert rc == 0
    assert set_config_calls == [("security.encryption.key_source", "passphrase")]
    assert keystore.primary_slot_type() == "passphrase"

    # The seeded .env must still decrypt under the freshly-cached DEK and
    # yield the original cleartext. (write_text() on Windows turns \n into
    # \r\n, so normalise before comparing rather than depending on host EOLs.)
    keystore.lock()
    keystore.unlock(passphrase="e2e-new-pw")
    plaintext = decrypt_if_encrypted(Path(env_path).read_bytes())
    assert plaintext.replace(b"\r\n", b"\n") == b"OPENAI_API_KEY=sk-test\n"

    # a successful real full re-key should have removed at least one
    # rekey-run backup (the seeded .env produced one), and the CLI must
    # surface that count in its output.
    captured = capsys.readouterr()
    assert "Cleaned" in captured.out
    assert "old-DEK backup(s)" in captured.out


def test_rotate_key_full_panel_prints_even_with_yes(monkeypatch, capsys):
    """the pre-flight panel must print even when ``--yes`` is set.

    Scripted operators running this from cron still need the
    DEK-replacement warning to land in the job log.
    """
    from hermes_cli import encrypt_cmd

    _seed_and_enable()

    def _stub_full_rekey(*_args, **_kwargs):
        return SimpleNamespace(
            rekeyed_files=[".env"],
            rekeyed_databases=[],
            rekeyed_sessions=[],
            rekeyed_logs=[],
            recovery_slots_dropped=0,
            skipped=[],
            backups_removed=0,
        )

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.full_rekey", _stub_full_rekey)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: "pw")

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args(yes=True))

    assert rc == 0
    captured = capsys.readouterr()
    # Panel header and the DEK-replacement warning are the load-bearing strings;
    # both must appear even though we passed --yes.
    assert "Full data-key re-key" in captured.out
    assert "new data encryption key" in captured.out


def test_rotate_key_full_surfaces_backups_removed(monkeypatch, capsys):
    """a successful real full re-key prints ``Cleaned N old-DEK backup(s)``."""
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    # No new key source — full re-key only. getpass returns the current pw.
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: "pw")

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args(yes=True))

    assert rc == 0
    captured = capsys.readouterr()
    assert "Cleaned" in captured.out
    assert "old-DEK backup(s)" in captured.out


# ---------------------------------------------------------------------------
# ``cmd_disable`` must accept a recovery code.
# The prior implementation bound ``recovery = None`` and never reassigned
# it; a user with a forgotten passphrase but a saved recovery code was
# locked out even though ``migrate.disable`` already supported the path.
# These tests pin the passphrase-mode recovery branch and the
# keyring/keyfile fallback that fires when the configured slot can no
# longer unlock (broken keyring, missing keyfile).
# ---------------------------------------------------------------------------


def _disable_args(**overrides) -> Namespace:
    base = {"yes": True, "force": True}
    base.update(overrides)
    return Namespace(**base)


def test_cmd_disable_accepts_recovery_code_in_passphrase_mode(monkeypatch):
    """User with a lost passphrase but saved recovery code can still disable.

    Mock flow: first getpass call (passphrase) returns ``""`` so cmd_disable
    falls through to the recovery prompt; the second getpass call returns
    the recovery code, which must reach ``migrate.disable`` unaltered.
    """
    from hermes_cli import encrypt_cmd

    _seed_and_enable(with_recovery=True)
    captured: dict = {}

    def _stub_disable(*, passphrase, recovery_code, **_kw):
        captured["passphrase"] = passphrase
        captured["recovery_code"] = recovery_code

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.disable", _stub_disable)
    answers = iter(["", "RECOVERY-CODE-ABCD-1234"])
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: next(answers)
    )

    rc = encrypt_cmd.cmd_disable(_disable_args())

    assert rc == 0
    assert captured["passphrase"] is None
    assert captured["recovery_code"] == "RECOVERY-CODE-ABCD-1234"


def test_cmd_disable_passphrase_path_unchanged(monkeypatch):
    """Existing 'type the passphrase' flow keeps its prior behavior unchanged."""
    from hermes_cli import encrypt_cmd

    _seed_and_enable(with_recovery=True)
    captured: dict = {}

    def _stub_disable(*, passphrase, recovery_code, **_kw):
        captured["passphrase"] = passphrase
        captured["recovery_code"] = recovery_code

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.disable", _stub_disable)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: "correct horse"
    )

    rc = encrypt_cmd.cmd_disable(_disable_args())

    assert rc == 0
    assert captured["passphrase"] == "correct horse"
    assert captured["recovery_code"] is None


def test_cmd_disable_no_passphrase_no_recovery_returns_error(monkeypatch):
    """Empty passphrase + empty recovery refuses to disable (exit 1)."""
    from hermes_cli import encrypt_cmd

    _seed_and_enable(with_recovery=True)
    called = {"disable": 0}

    def _fail_disable(**_kw):
        called["disable"] += 1
        raise AssertionError("migrate.disable must not run without credentials")

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.disable", _fail_disable)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: "")

    rc = encrypt_cmd.cmd_disable(_disable_args())

    assert rc == 1
    assert called["disable"] == 0


def test_cmd_disable_keyring_fallback_to_recovery(monkeypatch):
    """Broken keyring + recovery slot: prompt for the code and pass it through.

    Simulates a host where the keyring service is dead but the operator
    kept their recovery code. cmd_disable must catch the unlock failure,
    prompt for a recovery code, and forward it to ``migrate.disable``.
    """
    from hermes_cli import encrypt_cmd

    _seed_and_enable(with_recovery=True)
    captured: dict = {}

    def _stub_disable(*, passphrase, recovery_code, **_kw):
        captured["passphrase"] = passphrase
        captured["recovery_code"] = recovery_code

    def _broken_unlock(*_a, **_kw):
        raise RuntimeError("keyring service unavailable")

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.disable", _stub_disable)
    monkeypatch.setattr(
        "hermes_crypto.keystore.primary_slot_type", lambda: "keyring"
    )
    monkeypatch.setattr("hermes_crypto.keystore.has_recovery_slot", lambda: True)
    monkeypatch.setattr("hermes_crypto.keystore.unlock", _broken_unlock)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        encrypt_cmd.getpass, "getpass", lambda *_a, **_kw: "RECOVERY-FROM-KEYRING"
    )

    rc = encrypt_cmd.cmd_disable(_disable_args())

    assert rc == 0
    assert captured["passphrase"] is None
    assert captured["recovery_code"] == "RECOVERY-FROM-KEYRING"


def test_cmd_disable_keyring_no_recovery_unchanged(monkeypatch):
    """Keyring mode with no recovery slot: no prompt, ``migrate.disable`` handles unlock."""
    from hermes_cli import encrypt_cmd

    _seed_and_enable()  # no recovery slot
    captured: dict = {}
    getpass_called = {"n": 0}

    def _stub_disable(*, passphrase, recovery_code, **_kw):
        captured["passphrase"] = passphrase
        captured["recovery_code"] = recovery_code

    def _fail_getpass(*_a, **_kw):
        getpass_called["n"] += 1
        raise AssertionError(
            "getpass must not run in keyring mode without a recovery slot"
        )

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.disable", _stub_disable)
    monkeypatch.setattr(
        "hermes_crypto.keystore.primary_slot_type", lambda: "keyring"
    )
    monkeypatch.setattr("hermes_crypto.keystore.has_recovery_slot", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)

    rc = encrypt_cmd.cmd_disable(_disable_args())

    assert rc == 0
    assert captured["passphrase"] is None
    assert captured["recovery_code"] is None
    assert getpass_called["n"] == 0


# ── _prepare_new_key_source must prompt when passphrase ──
#    mode is rotating to itself. The original short-circuit returned
#    (True, None) and the downstream rotate_primary crashed with
#    ValueError("passphrase rotation requires a new passphrase").


def test_prepare_new_key_source_passphrase_to_passphrase_prompts_for_new(monkeypatch):
    """Regression: --key-source passphrase while current is passphrase
    must fall through to the new-passphrase prompt instead of short-circuiting.
    """
    from hermes_cli import encrypt_cmd
    from rich.console import Console

    prompts: list[str] = []

    def _fake_getpass(prompt: str) -> str:
        prompts.append(prompt)
        return "new-pw"

    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fake_getpass)

    console = Console()
    args = Namespace(key_source="passphrase")
    ok, new_passphrase = encrypt_cmd._prepare_new_key_source(
        console, args, current="passphrase"
    )

    assert ok is True
    assert new_passphrase == "new-pw"
    # Prompted twice (initial + confirm), not once and not zero.
    assert len(prompts) == 2, f"unexpected prompt count: {prompts}"


def test_prepare_new_key_source_keyring_to_keyring_skips_prompt(monkeypatch):
    """Regression: same-source rotation for non-passphrase modes still
    short-circuits (no fresh secret to collect). Pins that the passphrase
    fix doesn't accidentally widen to keyring/keyfile.
    """
    from hermes_cli import encrypt_cmd
    from rich.console import Console

    def _fail_getpass(*_a, **_k):
        raise AssertionError("keyring-to-keyring rotation must not prompt")

    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)
    monkeypatch.setattr(
        "hermes_crypto.keystore.keyring_is_secure", lambda: True
    )

    console = Console()
    args = Namespace(key_source="keyring")
    ok, new_passphrase = encrypt_cmd._prepare_new_key_source(
        console, args, current="keyring"
    )

    assert ok is True
    assert new_passphrase is None


def test_prepare_new_key_source_no_key_source_arg_short_circuits(monkeypatch):
    """Regression: when --key-source is not passed at all, no prompt
    fires and (True, None) is returned. Pins the baseline 'no change'
    behavior since the fix restructured the early-return logic.
    """
    from hermes_cli import encrypt_cmd
    from rich.console import Console

    def _fail_getpass(*_a, **_k):
        raise AssertionError("no --key-source flag must not prompt")

    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)

    console = Console()
    args = Namespace(key_source=None)
    ok, new_passphrase = encrypt_cmd._prepare_new_key_source(
        console, args, current="passphrase"
    )

    assert ok is True
    assert new_passphrase is None


# ---------------------------------------------------------------------------
# Non-TTY behavior: commands must never block on getpass when stdin is not a
# terminal — they either read HERMES_ENCRYPTION_PASSPHRASE or fail fast with
# an actionable error.
# ---------------------------------------------------------------------------


def _fail_getpass(*_a, **_k):
    raise AssertionError("getpass must never run when stdin is not a TTY")


def test_cmd_unlock_non_tty_without_env_fails_fast(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    keystore.lock()  # drop the cached DEK so unlock really runs
    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)
    monkeypatch.delenv("HERMES_ENCRYPTION_PASSPHRASE", raising=False)

    assert encrypt_cmd.cmd_unlock(Namespace()) == 1


def test_cmd_unlock_non_tty_uses_env_passphrase(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    keystore.lock()  # drop the cached DEK so unlock really runs
    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)
    monkeypatch.setenv("HERMES_ENCRYPTION_PASSPHRASE", "pw")

    assert encrypt_cmd.cmd_unlock(Namespace()) == 0


def test_cmd_disable_non_tty_uses_env_passphrase(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    keystore.lock()  # drop the cached DEK so unlock really runs
    captured: dict = {}

    def _stub_disable(*, passphrase, recovery_code, **_kw):
        captured["passphrase"] = passphrase
        captured["recovery_code"] = recovery_code

    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr("hermes_crypto.migrate.disable", _stub_disable)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)
    monkeypatch.setenv("HERMES_ENCRYPTION_PASSPHRASE", "pw")

    assert encrypt_cmd.cmd_disable(_disable_args()) == 0
    assert captured["passphrase"] == "pw"
    assert captured["recovery_code"] is None


def test_cmd_rotate_key_non_tty_without_env_fails_fast(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    keystore.lock()  # drop the cached DEK so unlock really runs
    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)
    monkeypatch.delenv("HERMES_ENCRYPTION_PASSPHRASE", raising=False)

    rc = encrypt_cmd.cmd_rotate_key(_full_rotate_args(yes=True))

    assert rc == 1


def test_cmd_add_recovery_non_tty_without_env_fails_fast(monkeypatch):
    from hermes_cli import encrypt_cmd

    _seed_and_enable()
    keystore.lock()  # drop the cached DEK so unlock really runs
    monkeypatch.setattr(encrypt_cmd, "_ensure_deps", lambda *_a, **_k: True)
    monkeypatch.setattr(encrypt_cmd.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(encrypt_cmd.getpass, "getpass", _fail_getpass)
    monkeypatch.delenv("HERMES_ENCRYPTION_PASSPHRASE", raising=False)

    assert encrypt_cmd.cmd_add_recovery(Namespace()) == 1

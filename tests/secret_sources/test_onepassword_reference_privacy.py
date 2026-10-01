"""Configured vault locators must not be echoed by secret-resolution failures."""
import subprocess

import pytest

from agent.secret_sources import onepassword as op


@pytest.mark.parametrize("failure", ["stderr", "empty", "timeout", "spawn", "exit", "ansi", "quoted"])
def test_fetch_hides_reference_but_names_destination(monkeypatch, tmp_path, failure):
    reference = "op://Private Vault/Client's account/api key"
    if failure == "quoted":
        reference += ' "backup"'

    def run(cmd, **kwargs):
        assert cmd[-1] == reference
        assert cmd[cmd.index("--account") + 1] == "chosen-account"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 30)
        if failure == "spawn":
            raise OSError(f"cannot invoke {reference}")
        if failure == "empty":
            return subprocess.CompletedProcess(cmd, 0, " \n", "")
        diagnostic = f"not signed in: {reference!r}"
        if failure == "ansi":
            diagnostic = "not signed in: " + reference.replace("Vault", "\x1b[31mVault\x1b[0m")
        return subprocess.CompletedProcess(
            cmd, 1, "", diagnostic if failure in {"stderr", "ansi", "quoted"} else ""
        )

    monkeypatch.setattr(op.subprocess, "run", run)
    secrets, warnings = op.fetch_onepassword_secrets(
        references={"TARGET_KEY": reference}, binary=tmp_path / "op",
        account="chosen-account", use_cache=False, home_path=tmp_path,
    )
    assert secrets == {}
    assert len(warnings) == 1
    assert reference not in warnings[0]
    assert "Private Vault" not in warnings[0]
    assert "TARGET_KEY" in warnings[0]
    if failure == "stderr":
        assert "not signed in" in warnings[0]


def test_invalid_reference_value_is_not_echoed(monkeypatch, tmp_path):
    reference = "mistyped://Private Vault/Client account/password"
    secrets, warnings = op.fetch_onepassword_secrets(
        references={"TARGET_KEY": reference}, binary=tmp_path / "op",
        use_cache=False, home_path=tmp_path,
    )
    assert secrets == {}
    assert reference not in " ".join(warnings)
    assert "TARGET_KEY" in " ".join(warnings)


def test_reference_masking_precedes_truncation_and_preserves_partial_success(monkeypatch, tmp_path):
    bad = "op://" + "private-vault-" * 30 + "/item/field"
    good = "op://Other Vault/other item/field"

    def run(cmd, **kwargs):
        if cmd[-1] == good:
            return subprocess.CompletedProcess(cmd, 0, "  usable value  \n", "")
        return subprocess.CompletedProcess(cmd, 1, "", f"not signed in: {bad}; related {good}")

    monkeypatch.setattr(op.subprocess, "run", run)
    secrets, warnings = op.fetch_onepassword_secrets(
        references={"GOOD_KEY": good, "BAD_KEY": bad}, binary=tmp_path / "op",
        use_cache=False, home_path=tmp_path,
    )
    assert secrets == {"GOOD_KEY": "  usable value  "}
    assert len(warnings) == 1
    assert "private-vault" not in warnings[0]
    assert "Other Vault" not in warnings[0]
    assert "not signed in" in warnings[0]

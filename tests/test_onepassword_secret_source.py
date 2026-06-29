"""Tests for the 1Password secret-source backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import onepassword  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache():
    onepassword._reset_cache_for_tests()
    yield
    onepassword._reset_cache_for_tests()


def test_parse_reference_file_accepts_only_op_references(tmp_path):
    env_file = tmp_path / "1password.env"
    env_file.write_text(
        "# comment\n"
        "OPENROUTER_API_KEY=op://Private/OpenRouter API Key/credential\n"
        "export RUNWARE_API_KEY='op://Private/Runware API Key/credential'\n"
        "PLAINTEXT_API_KEY=sk-plaintext\n"
        "1BAD=op://Private/Bad/credential\n"
        "BROKEN_LINE\n",
        encoding="utf-8",
    )

    refs, warnings = onepassword.parse_reference_file(env_file)

    assert refs == {
        "OPENROUTER_API_KEY": "op://Private/OpenRouter API Key/credential",
        "RUNWARE_API_KEY": "op://Private/Runware API Key/credential",
    }
    assert len(warnings) == 3
    assert any("PLAINTEXT_API_KEY" in warning for warning in warnings)
    assert any("1BAD" in warning for warning in warnings)
    assert any("expected NAME=op://" in warning for warning in warnings)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fake executable uses POSIX shebang/chmod")
def test_fetch_onepassword_secrets_resolves_references_with_op_read(tmp_path):
    env_file = tmp_path / "1password.env"
    env_file.write_text(
        "OPENROUTER_API_KEY=op://Private/OpenRouter/credential\n"
        "RUNWARE_API_KEY=op://Private/Runware/credential\n"
        "MISSING_API_KEY=op://Private/Missing/credential\n",
        encoding="utf-8",
    )
    fake_op = tmp_path / "op"
    fake_op.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1] == '--version':\n"
        "    print('2.30.0')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1] == 'read':\n"
        "    ref = sys.argv[2]\n"
        "    values = {\n"
        "        'op://Private/OpenRouter/credential': 'test-openrouter-secret',\n"
        "        'op://Private/Runware/credential': 'test-runware-secret',\n"
        "    }\n"
        "    if ref not in values:\n"
        "        print('could not read op://Private/Missing/credential', file=sys.stderr)\n"
        "        raise SystemExit(1)\n"
        "    print(values[ref])\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    fake_op.chmod(0o700)

    secrets, warnings = onepassword.fetch_onepassword_secrets(
        env_file=str(env_file),
        binary=fake_op,
        use_cache=False,
    )

    assert secrets == {
        "OPENROUTER_API_KEY": "test-openrouter-secret",
        "RUNWARE_API_KEY": "test-runware-secret",
    }
    assert len(warnings) == 1
    assert "MISSING_API_KEY" in warnings[0]
    assert "op://[REDACTED]" in warnings[0]
    assert "op://Private/Missing/credential" not in warnings[0]


def test_apply_onepassword_secrets_respects_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / "1password.env"
    env_file.write_text("OPENROUTER_API_KEY=op://Private/OpenRouter/credential\n", encoding="utf-8")
    fake_op = tmp_path / "op"
    fake_op.write_text("placeholder", encoding="utf-8")
    fake_op.chmod(0o700)

    monkeypatch.setenv("OPENROUTER_API_KEY", "existing")
    monkeypatch.setattr(onepassword, "find_op", lambda *, op_path="": fake_op)
    monkeypatch.setattr(
        onepassword,
        "fetch_onepassword_secrets",
        lambda **_kwargs: ({"OPENROUTER_API_KEY": "from-1password"}, []),
    )

    result = onepassword.apply_onepassword_secrets(
        enabled=True,
        env_file=str(env_file),
        override_existing=False,
    )

    assert result.applied == []
    assert result.skipped == ["OPENROUTER_API_KEY"]
    assert os.environ["OPENROUTER_API_KEY"] == "existing"

    result = onepassword.apply_onepassword_secrets(
        enabled=True,
        env_file=str(env_file),
        override_existing=True,
    )

    assert result.applied == ["OPENROUTER_API_KEY"]
    assert os.environ["OPENROUTER_API_KEY"] == "from-1password"


def test_apply_onepassword_secrets_missing_op_is_nonfatal():
    result = onepassword.apply_onepassword_secrets(
        enabled=True,
        env_file="/does/not/matter.env",
        op_path="/definitely/missing/op",
    )

    assert result.applied == []
    assert result.error
    assert "op" in result.error


def test_default_references_file_respects_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    resolved = onepassword._resolve_env_file("~/.hermes/secrets/1password.env")

    assert resolved == tmp_path / "secrets" / "1password.env"

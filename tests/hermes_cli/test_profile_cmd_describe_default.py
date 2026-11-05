"""``hermes profile describe``: the ``default`` alias addresses the install root.

A multiplexed gateway hosts a CLI session in ``~/.hermes/profiles/<active>``, so
``HERMES_HOME`` — and therefore ``get_hermes_home()`` — is that satellite profile's
directory. ``default`` is not a directory name: it is the alias for the install root
(``get_profile_dir("default")`` → ``get_default_hermes_root()``), which is the value
``read_profile_meta`` / the roster / the web routers already use. Resolving it through
``get_hermes_home()`` made both the read and the write land on the calling profile, so
``describe default --text`` silently overwrote the caller's own description.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import profile_cmd
from hermes_cli import profile_describer
from hermes_cli import profiles as profiles_mod


@pytest.fixture
def multi_profile_install(tmp_path, monkeypatch):
    """Install root + one satellite profile, with the CLI hosted by the satellite."""
    root = tmp_path / ".hermes"
    coach = root / "profiles" / "coach"
    coach.mkdir(parents=True)
    (root / "config.yaml").write_text('model: {provider: custom, default: m}\n', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(coach))
    profiles_mod.write_profile_meta(root, description="root description", description_auto=False)
    profiles_mod.write_profile_meta(coach, description="coach description", description_auto=False)
    return root, coach


def _describe_args(**overrides):
    args = {"profile_name": "default", "text": None, "auto": False, "all_missing": False, "overwrite": False}
    args.update(overrides)
    return SimpleNamespace(**args)


def test_describe_target_dir_maps_default_to_the_install_root(multi_profile_install):
    root, coach = multi_profile_install
    assert profile_cmd._describe_target_dir("default") == root
    assert profile_cmd._describe_target_dir("Default") == root
    # Named profiles keep resolving to their own directory.
    assert profile_cmd._describe_target_dir("coach") == coach


def test_describe_default_reads_the_install_roots_description(multi_profile_install, capsys):
    with pytest.raises(SystemExit) as exit_info:
        profile_cmd._profile_describe(_describe_args())
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "root description"


def test_describe_default_text_writes_the_install_root(multi_profile_install, capsys):
    root, coach = multi_profile_install
    with pytest.raises(SystemExit) as exit_info:
        profile_cmd._profile_describe(_describe_args(text="cross-domain orchestration"))
    assert exit_info.value.code == 0
    assert "Description updated for 'default'." in capsys.readouterr().out
    assert profiles_mod.read_profile_meta(root)["description"] == "cross-domain orchestration"
    # The calling profile's own description is untouched.
    assert profiles_mod.read_profile_meta(coach)["description"] == "coach description"


def test_describe_auto_default_writes_the_install_root(multi_profile_install):
    root, coach = multi_profile_install
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = '{"description": "nightly digests and batch scheduling"}'
    with patch("agent.auxiliary_client.call_llm", return_value=resp):
        outcome = profile_describer.describe_profile("default", overwrite=True)
    assert outcome.ok, outcome.reason
    meta = profiles_mod.read_profile_meta(root)
    assert meta["description"] == "nightly digests and batch scheduling"
    assert meta["description_auto"] is True
    assert profiles_mod.read_profile_meta(coach)["description"] == "coach description"

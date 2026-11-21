"""Paper's Desktop relay uses a required non-secret executable, not credentials."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import mcp_catalog
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home


@pytest.fixture
def paper_entry(monkeypatch):
    catalog = Path(__file__).parents[2] / "optional-mcps"
    monkeypatch.setattr(mcp_catalog, "_catalog_root", lambda: catalog)
    entry = mcp_catalog.get_entry("paper")
    assert entry is not None
    return entry


@pytest.fixture
def no_credentials(monkeypatch):
    for name in ("get_env_value", "save_env_value"):
        monkeypatch.setattr(
            mcp_catalog, name, Mock(side_effect=AssertionError("credential access"))
        )


@pytest.mark.parametrize("enable", [False, True])
def test_paper_install_keeps_executable_with_spaces_as_one_command(
    paper_entry, no_credentials, tmp_path, monkeypatch, enable
):
    executable = str(tmp_path / "Paper Desktop" / "paper")
    prompt = Mock(return_value=executable)
    monkeypatch.setattr(mcp_catalog, "_prompt_input", prompt)
    probe = Mock(return_value=None)
    monkeypatch.setattr(mcp_catalog, "_probe_tools", probe)

    mcp_catalog.install_entry(paper_entry, enable=enable)

    prompt.assert_called_once_with(
        paper_entry.auth.env[0].prompt, default=None, password=False
    )
    assert paper_entry.auth.type == "none"
    assert paper_entry.install is None
    assert load_config()["mcp_servers"]["paper"] == {
        "command": executable, "args": ["mcp"], "enabled": enable
    }
    assert not (get_hermes_home() / ".env").exists()
    probe.assert_called_once_with("paper")


def test_paper_requires_executable_before_saving_or_probing(
    paper_entry, no_credentials, monkeypatch
):
    from hermes_cli import mcp_config

    monkeypatch.setattr(mcp_catalog, "_prompt_input", Mock(return_value=""))
    save = Mock()
    probe = Mock()
    monkeypatch.setattr(mcp_config, "_save_mcp_server", save)
    monkeypatch.setattr(mcp_catalog, "_probe_tools", probe)

    with pytest.raises(mcp_catalog.CatalogError, match="PAPER_CLI_PATH is required"):
        mcp_catalog.install_entry(paper_entry)

    save.assert_not_called()
    probe.assert_not_called()

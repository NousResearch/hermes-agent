"""Write-back requires a mapping in an existing config document (regression for #147350)."""

from types import SimpleNamespace

import pytest

from hermes_cli import config
from hermes_state_runtime import RuntimeStoreError


_DOCUMENTS = [
    pytest.param("", False, id="empty"),
    pytest.param("# interrupted edit\n", False, id="comment-only"),
    pytest.param("null\n", False, id="null"),
    pytest.param("[]\n", False, id="list"),
    pytest.param("false\n", False, id="boolean"),
    pytest.param("unfinished\n", False, id="scalar"),
    pytest.param(None, True, id="absent"),
    pytest.param("{}\n", True, id="empty-mapping"),
    pytest.param("# keep\ncustom: ${MY_SETTING}\n", True, id="mapping"),
]


@pytest.mark.parametrize("document,readable", _DOCUMENTS)
@pytest.mark.parametrize("read", [config.read_user_config_raw, config.require_readable_config_before_write])
def test_writeback_readers_require_an_existing_mapping(tmp_path, document, readable, read):
    path = tmp_path / "config.yaml"
    if document is not None:
        path.write_text(document, encoding="utf-8")

    if readable:
        assert read(path) == ({"custom": "${MY_SETTING}"} if document and "custom:" in document else {})
    else:
        with pytest.raises(RuntimeError, match="must be a mapping"):
            read(path)
        assert path.read_text(encoding="utf-8") == document


def _raw_roundtrip(path):
    data = config.read_user_config_raw(path)
    data.setdefault("display", {})["message_reactions"] = True
    config.atomic_config_replace(path, data)


def _gateway_roundtrip(path):
    from gateway.session_tui_config import _write_profile_config

    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(path.parent)))
    _write_profile_config(connection, lambda data: data.setdefault("display", {}).update(message_reactions=True))


@pytest.mark.parametrize("document,readable", _DOCUMENTS)
@pytest.mark.parametrize("write", [_raw_roundtrip, _gateway_roundtrip])
def test_profile_setting_write_preserves_unusable_documents(tmp_path, document, readable, write):
    path = tmp_path / "config.yaml"
    if document is not None:
        path.write_text(document, encoding="utf-8")

    if readable:
        write(path)
        saved = config.read_user_config_raw(path)
        assert saved["display"]["message_reactions"] is True
        if document and "custom:" in document:
            assert saved["custom"] == "${MY_SETTING}"
            assert "# keep" in path.read_text(encoding="utf-8")
    else:
        with pytest.raises((RuntimeError, RuntimeStoreError)):
            write(path)
        assert path.read_text(encoding="utf-8") == document

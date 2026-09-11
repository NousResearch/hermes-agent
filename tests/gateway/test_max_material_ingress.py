"""Max material ingress rejects unsafe Telegram identifiers before staging."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock


MODULE = Path(__file__).parents[2] / "gateway" / "max_material_ingress.py"
SPEC = importlib.util.spec_from_file_location("max_material_ingress_under_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
INGRESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INGRESS
SPEC.loader.exec_module(INGRESS)


def test_unsafe_media_group_identifier_never_creates_a_staging_path(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(INGRESS.subprocess, "run", run)
    image = tmp_path / "word.jpg"
    image.write_bytes(b"image")
    staging = tmp_path / "staging"

    result = INGRESS.dispatch(
        settings={"enabled": True, "chat_id": "1", "user_id": "2", "staging_root": str(staging)},
        chat_id="1", user_id="2", message_ids=["123"], media_group_id="../../outside",
        cached_paths=[str(image)],
    )

    assert result.handled is True
    assert result.status == "failed"
    assert not staging.exists()
    run.assert_not_called()


def test_safe_telegram_identifiers_reach_the_fixed_wrapper(tmp_path, monkeypatch):
    run = Mock(return_value=Mock(returncode=1))
    monkeypatch.setattr(INGRESS.subprocess, "run", run)
    image = tmp_path / "word.jpg"
    image.write_bytes(b"image")
    staging = tmp_path / "staging"

    result = INGRESS.dispatch(
        settings={"enabled": True, "chat_id": "1", "user_id": "2", "staging_root": str(staging)},
        chat_id="1", user_id="2", message_ids=["123"], media_group_id="album_456-7",
        cached_paths=[str(image)],
    )

    assert result.status == "retrying"
    assert (staging / "album-album_456-7").is_dir()
    run.assert_called_once()

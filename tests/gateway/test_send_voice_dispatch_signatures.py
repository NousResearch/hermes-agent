"""Every platform adapter's ``send_voice`` must accept the ``is_voice`` keyword the base media
dispatch passes (``BasePlatformAdapter._deliver_media_from_response`` →
``send_voice(..., is_voice=is_voice)``). An explicit signature without it raises ``TypeError`` and
the attachment is silently dropped (#102221, #116776 — Matrix; same class in line/mattermost/weixin)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_FILES = sorted(
    p for p in list((_ROOT / "plugins" / "platforms").glob("*/adapter.py")) + list((_ROOT / "gateway" / "platforms").glob("*.py"))
    if "async def send_voice" in p.read_text(encoding="utf-8")
)


def _send_voice_defs(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "send_voice"]


@pytest.mark.parametrize("path", _ADAPTER_FILES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_every_send_voice_accepts_is_voice(path: Path) -> None:
    for fn in _send_voice_defs(path):
        names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        assert fn.args.kwarg is not None or "is_voice" in names, (
            f"{path}: send_voice() rejects the dispatch's is_voice kwarg — add **kwargs or is_voice"
        )


def test_base_dispatch_passes_is_voice_to_send_voice() -> None:
    """Guards the contract from the other side: the dispatch really passes ``is_voice``."""
    from gateway.platforms.base import BasePlatformAdapter

    src = inspect.getsource(BasePlatformAdapter)
    assert "self.send_voice(chat_id=chat_id, audio_path=path, metadata=metadata, is_voice=is_voice)" in src

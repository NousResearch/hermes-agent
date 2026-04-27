"""Static dashboard tests for browser-safe @nous-research/ui imports."""
from pathlib import Path


WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"


def test_dashboard_does_not_import_nous_ui_root_barrel():
    offenders = []
    for path in WEB_SRC.rglob("*.tsx"):
        content = path.read_text(encoding="utf-8")
        if 'from "@nous-research/ui"' in content or "from '@nous-research/ui'" in content:
            offenders.append(str(path.relative_to(WEB_SRC)))

    assert offenders == []

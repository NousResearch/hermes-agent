"""stamp() also refreshes hermes_cli.__release_date__ so the date shown by
``hermes --version`` tracks the release being stamped (issue #148353)."""
from pathlib import Path


def _hermes_init(tree: Path) -> Path:
    init = tree / "hermes_cli" / "__init__.py"
    init.parent.mkdir(parents=True, exist_ok=True)
    if not init.exists():
        init.write_text('__release_date__ = "2026.9.24"\n', encoding="utf-8")
    return init


def test_stamp_refreshes_release_date(tmp_path):
    from scripts.releases.stamping import stamp

    init = _hermes_init(tmp_path)
    stamp(tmp_path, "0.22.5", release_date="2026.10.23")
    assert init.read_text(encoding="utf-8").strip() == '__release_date__ = "2026.10.23"'


def test_stamp_without_release_date_leaves_literal_untouched(tmp_path):
    from scripts.releases.stamping import stamp

    init = _hermes_init(tmp_path)
    stamp(tmp_path, "0.22.6")
    assert init.read_text(encoding="utf-8").strip() == '__release_date__ = "2026.9.24"'

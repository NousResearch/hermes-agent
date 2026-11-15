"""A plugin that sends credentials through ``open_credentialed_url`` — the redirect-safe opener the
plugin docs point to — gets no private-import warning from ``hermes plugins validate``."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml

from hermes_cli.plugin_validate import validate_plugin_dir


def test_open_credentialed_url_import_is_public(tmp_path: Path):
    root = tmp_path / "fetcher"
    root.mkdir()
    (root / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "fetcher", "version": "1.0.0", "description": "x"}), encoding="utf-8")
    (root / "__init__.py").write_text(
        "from hermes_cli.urllib_security import open_credentialed_url\n"
        "def register(ctx):\n"
        "    pass\n", encoding="utf-8")

    report = validate_plugin_dir(root)

    assert [w for w in report.warnings if w.startswith("private core imports")] == []

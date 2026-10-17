"""Regression for #134965: the workspace snapshot must carry the bundled
``locales/`` directory. It is i18n data (``agent/i18n.py::_locales_dir``
resolves ``<repo-root>/locales``), not a setuptools package, so the old
package-roots-only copy list silently dropped it from every PM-materialized
environment and ``t()`` fell back to bare dotted keys."""

from pathlib import Path

from pm import workspace


def test_copy_core_inputs_carries_bundled_locales(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="replay-plugin"\nversion="1.0"\n'
        '[tool.setuptools.packages.find]\ninclude=["replay_plugin*"]\n', encoding="utf-8"
    )
    (source / "replay_plugin").mkdir()
    (source / "replay_plugin/__init__.py").write_text("", encoding="utf-8")
    (source / "locales").mkdir()
    (source / "locales/en.yaml").write_text("cli:\n  tip_line: tip\n", encoding="utf-8")
    destination = tmp_path / "dest"
    destination.mkdir()

    workspace._copy_core_inputs(source, destination)

    assert (destination / "locales/en.yaml").is_file()
    assert Path(destination / "locales/en.yaml").read_text(encoding="utf-8") == (
        "cli:\n  tip_line: tip\n"
    )


def test_copy_core_inputs_still_skips_non_package_dirs(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="replay-plugin"\nversion="1.0"\n', encoding="utf-8"
    )
    (source / "replay_plugin").mkdir()
    (source / "replay_plugin/__init__.py").write_text("", encoding="utf-8")
    for stray in ("build", "release"):
        (source / stray).mkdir()
        (source / stray / "artifact.bin").write_text("x", encoding="utf-8")
    destination = tmp_path / "dest"
    destination.mkdir()

    workspace._copy_core_inputs(source, destination)

    assert not (destination / "build").exists()
    assert not (destination / "release").exists()

"""Regression coverage for the Termux manual install guide."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TERMUX_DOCS = [
    REPO_ROOT / "website" / "docs" / "getting-started" / "termux.md",
    REPO_ROOT
    / "website"
    / "i18n"
    / "zh-Hans"
    / "docusaurus-plugin-content-docs"
    / "current"
    / "getting-started"
    / "termux.md",
]


def test_manual_termux_install_runs_psutil_shim_before_pip_install() -> None:
    """The documented manual path must match install.sh's Android psutil prebuild."""
    for doc in TERMUX_DOCS:
        text = doc.read_text(encoding="utf-8")
        manual_start = (
            text.find("## Option 2: Manual install")
            if "## Option 2: Manual install" in text
            else text.index("## 方式二：手动安装")
        )
        manual_text = text[manual_start:]
        shim_idx = manual_text.index("python scripts/install_psutil_android.py")
        install_idx = manual_text.index("python -m pip install -e '.[termux]'")

        assert shim_idx < install_idx, f"{doc} installs package before psutil shim"


def test_termux_docs_cover_32_bit_maturin_target() -> None:
    """Termux troubleshooting should cover the armv8l/maturin failure from #31415."""
    for doc in TERMUX_DOCS:
        text = doc.read_text(encoding="utf-8")

        assert "Unsupported Android architecture: armv8l" in text
        assert 'CARGO_BUILD_TARGET="armv7-linux-androideabi"' in text
        assert 'CARGO_BUILD_TARGET="aarch64-linux-android"' in text

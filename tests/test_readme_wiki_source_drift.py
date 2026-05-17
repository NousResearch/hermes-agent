from __future__ import annotations

import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _toolset_names() -> set[str]:
    text = _read("toolsets.py")
    match = re.search(r"TOOLSETS\s*=\s*\{(?P<body>.*?)(?:\n\}\n)", text, re.S)
    if not match:
        raise AssertionError("TOOLSETS mapping not found")
    return set(re.findall(r"^[ \t]*[\"']([^\"']+)[\"']\s*:", match.group("body"), re.M))


def _registered_tool_count() -> int:
    total = 0
    for path in (REPO_ROOT / "tools").glob("*.py"):
        total += len(re.findall(r"registry\.register\s*\(", path.read_text(encoding="utf-8")))
    return total


def test_readmes_match_current_windows_install_and_terminal_backend_facts() -> None:
    readme = _read("README.md")
    readme_zh = _read("README.zh-CN.md")

    assert "PortableGit" in readme
    assert "Node.js 22" in readme
    assert "Native Windows support is **early beta**" in readme

    assert "NovitaAI" in readme_zh
    assert "七种终端后端" in readme_zh
    assert "Vercel Sandbox" in readme_zh
    assert "install.ps1" in readme_zh
    assert "原生 Windows 不受支持" not in readme_zh
    assert "六种终端后端" not in readme_zh


def test_contributor_docs_reference_real_gateway_platform_files_and_test_runner() -> None:
    contributing = _read("CONTRIBUTING.md")
    website_contributing = _read("website/docs/developer-guide/contributing.md")

    assert (REPO_ROOT / "gateway/platforms/discord.py").exists()
    assert "discord.py" in contributing
    assert "discord_adapter.py" not in contributing

    assert (REPO_ROOT / "scripts/run_tests.sh").exists()
    assert "scripts/run_tests.sh" in contributing
    assert "scripts/run_tests.sh" in website_contributing


def test_architecture_docs_use_current_scale_claims_without_stale_exact_counts() -> None:
    architecture = _read("website/docs/developer-guide/architecture.md")
    agents = _read("AGENTS.md")

    assert _registered_tool_count() >= 70
    assert len(_toolset_names()) >= 50
    assert "70+ statically registered tools across 50+ toolsets" in architecture
    assert "~28 toolsets" not in architecture

    assert len(list((REPO_ROOT / "tests").rglob("test_*.py"))) >= 1000
    assert "1,000+ test files" in agents
    assert "~17k tests" not in agents

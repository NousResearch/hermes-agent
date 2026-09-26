"""Tests for optional-skills/productivity/memory-extension.

Covers the SKILL.md contract and the check-memory-coherence.sh script logic:
orphan detection, dangling-reference detection, and clean-state exit.
Also covers the contradiction-detection scripts: deterministic pass 1
(duplicate keys, negations, versions, dates) and LLM pass 2 (dry-run only,
no network).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "memory-extension"
SCRIPT = SKILL_DIR / "scripts" / "check-memory-coherence.sh"
SCRIPT_CONTRADICT = SKILL_DIR / "scripts" / "check-memory-contradictions.sh"
SCRIPT_LLM = SKILL_DIR / "scripts" / "check-memory-contradictions-llm.py"


def _bash_binary():
    """Return a working bash binary.

    On Windows, plain `bash` may resolve to the WSL relay (/usr/bin/bash),
    which fails when invoked from a subprocess with a Windows cwd. Prefer
    git-bash explicitly; fall back to PATH `bash` (Linux/macOS CI).
    """
    if sys.platform == "win32":
        candidates = [
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    found = shutil.which("bash")
    if found and "wsl" not in str(found).lower():
        return found
    return None


BASH = _bash_binary()


def _frontmatter():
    content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    return yaml.safe_load(content[3 : m.start() + 3]), content


def _run_script(home: Path) -> subprocess.CompletedProcess:
    assert BASH, "no working bash binary found"
    return subprocess.run(
        [BASH, str(SCRIPT), str(home)],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestFrontmatter:
    def test_required_fields(self):
        fm, _ = _frontmatter()
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"

    def test_name_matches_directory(self):
        fm, _ = _frontmatter()
        assert fm["name"] == SKILL_DIR.name

    def test_description_hardline(self):
        fm, _ = _frontmatter()
        desc = str(fm.get("description") or "")
        assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
        assert desc.rstrip().endswith(".")

    def test_platforms_cover_script_shell(self):
        fm, _ = _frontmatter()
        # the script is bash; Windows is covered via git-bash/MSYS
        assert "windows" in fm.get("platforms", [])

    def test_skill_references_existing_files(self):
        _, content = _frontmatter()
        refs = re.findall(r"references/([A-Za-z0-9_-]+\.md)", content)
        refs += re.findall(r"scripts/([A-Za-z0-9_-]+\.sh)", content)
        assert refs, "SKILL.md should reference its references/ and scripts/"
        for ref in refs:
            candidates = list((SKILL_DIR / "references").glob(ref)) + list(
                (SKILL_DIR / "scripts").glob(ref)
            )
            assert candidates, f"SKILL.md references missing file: {ref}"


@pytest.fixture()
def fake_home(tmp_path):
    """Builds a coherent extended-memory layout, returns (home, write_fn)."""
    home = tmp_path / "hermes"
    memories = home / "memories"
    (memories / "extended").mkdir(parents=True)
    (memories / "MEMORY.md").write_text(
        "Topic A — hint → see extended/topic-a.md\n", encoding="utf-8"
    )
    (memories / "USER.md").write_text("", encoding="utf-8")
    (memories / "extended" / "topic-a.md").write_text("# Topic A\n", encoding="utf-8")
    (memories / "extended" / "README.md").write_text("# guide\n", encoding="utf-8")
    return home


@pytest.mark.skipif(BASH is None, reason="no working bash binary found")
class TestCoherenceScript:
    def test_clean_state_exits_zero(self, fake_home):
        r = _run_script(fake_home)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "all referenced" in r.stdout

    def test_detects_orphan_file(self, fake_home):
        (fake_home / "memories" / "extended" / "orphan.md").write_text(
            "# orphan\n", encoding="utf-8"
        )
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "orphan.md" in r.stdout

    def test_detects_dangling_reference(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "Topic A — hint → see extended/topic-a.md\n"
            "Ghost — hint → see extended/ghost.md\n",
            encoding="utf-8",
        )
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "ghost.md" in r.stdout

    def test_ignores_readme(self, fake_home):
        # README.md is the guide copy; it must not count as an orphan
        r = _run_script(fake_home)
        assert r.returncode == 0
        assert "README.md" not in r.stdout

    def test_detects_dangling_multidot_reference(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "Topic A — hint → see extended/topic-a.md\n"
            "Ghost — hint → see extended/ghost.bar.md\n",
            encoding="utf-8",
        )
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "ghost.bar.md" in r.stdout

    def test_missing_index_reports_error(self, fake_home):
        (fake_home / "memories" / "MEMORY.md").unlink()
        (fake_home / "memories" / "USER.md").unlink()
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "no index" in r.stdout.lower()


def _run_contradict(home: Path) -> subprocess.CompletedProcess:
    assert BASH, "no working bash binary found"
    return subprocess.run(
        [BASH, str(SCRIPT_CONTRADICT), str(home)],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.skipif(BASH is None, reason="no working bash binary found")
class TestContradictionScript:
    def test_clean_state_exits_zero(self, fake_home):
        r = _run_contradict(fake_home)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "no contradiction candidate" in r.stdout.lower()

    def test_detects_duplicate_key_different_values(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "Model default: deepseek-flash\n"
            "Model default: claude-sonnet\n",
            encoding="utf-8",
        )
        r = _run_contradict(fake_home)
        assert r.returncode == 1
        assert "duplicate key" in r.stdout.lower()

    def test_detects_strong_negation(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "LM Studio: never launch it.\n",
            encoding="utf-8",
        )
        r = _run_contradict(fake_home)
        assert r.returncode == 1
        assert "strong negation" in r.stdout.lower()

    def test_detects_multiple_versions(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "ComfyUI v0.33.1 installed\n"
            "ComfyUI v0.32.0 installed\n",
            encoding="utf-8",
        )
        r = _run_contradict(fake_home)
        assert r.returncode == 1
        assert "multiple versions" in r.stdout.lower()

    def test_detects_multiple_dates(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "Event on 01/01/2026 and 02/02/2026\n",
            encoding="utf-8",
        )
        r = _run_contradict(fake_home)
        assert r.returncode == 1
        assert "multiple dates" in r.stdout.lower()


class TestContradictionScriptLocaleGuard:
    """Regression guard: the [A-Za-zÀ-ÿ0-9] ranges in the deterministic
    script break grep on UTF-8 locales (\"grep: invalid range end\"), so the
    script must force LC_ALL=C. Reported from a fr_FR.UTF-8 VM (res-89)."""

    def test_script_forces_c_locale(self):
        script = (SKILL_DIR / "scripts" / "check-memory-contradictions.sh").read_text(encoding="utf-8")
        assert "export LC_ALL=C" in script, "script must force LC_ALL=C for UTF-8 locales"

    def test_multibyte_ranges_used_after_locale_guard(self):
        """The multibyte [..À-ÿ..] ranges stay in the code (they match accented
        names), and the LC_ALL=C guard must execute before them so grep treats
        them byte-wise on UTF-8 locales. Occurrences in the guard comment above
        the export are harmless."""
        script = (SKILL_DIR / "scripts" / "check-memory-contradictions.sh").read_text(encoding="utf-8")
        guard_at = script.find("export LC_ALL=C")
        assert guard_at != -1, "script must force LC_ALL=C for UTF-8 locales"
        assert script.find("À-ÿ", guard_at) != -1, "no À-ÿ range found after the guard"


class TestContradictionLLMScript:
    def test_dry_run_no_network(self, fake_home):
        """--dry-run must not call the API and must exit 0."""
        r = subprocess.run(
            [sys.executable, str(SCRIPT_LLM), str(fake_home), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "dry-run" in r.stdout.lower()

    def test_missing_memory_reports_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        r = subprocess.run(
            [sys.executable, str(SCRIPT_LLM), str(empty), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode != 0
        assert "no memory" in (r.stdout + r.stderr).lower()

    def test_parse_api_response_accepts_json(self):
        """parse_api_response returns a dict for a valid JSON body."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("mem_llm", SCRIPT_LLM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        body = '{"choices": [{"message": {"content": "[]"}}]}'
        assert mod.parse_api_response(body)["choices"][0]["message"]["content"] == "[]"

    def test_parse_api_response_rejects_non_json(self):
        """parse_api_response raises JSONDecodeError on a plain-text body
        (e.g. a server-latency page) so the caller can retry."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("mem_llm2", SCRIPT_LLM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with pytest.raises(json.JSONDecodeError):
            mod.parse_api_response("<html>524 origin timeout</html>")

    def test_extract_json_array_chatty_reply(self):
        """extract_json_array finds the JSON array inside a chatty reply."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("mem_llm3", SCRIPT_LLM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        reply = (
            "Some reflection before..." +
                '[{"fait_a": "A", "source_a": "MEMORY.md", "fait_b": "B", "source_b": "USER.md", "raison": "R"}]' +
            "And reflection after."
        )
        out = mod.extract_json_array(reply)
        assert isinstance(out, list) and len(out) == 1
        assert out[0]["fait_a"] == "A"

    def test_extract_json_array_fenced(self):
        """extract_json_array handles a fenced ```json block and empty []."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("mem_llm4", SCRIPT_LLM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.extract_json_array("```json\n[]\n```") == []
        assert mod.extract_json_array("no json here") is None


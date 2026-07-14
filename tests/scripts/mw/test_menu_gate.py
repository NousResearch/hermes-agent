"""Hermetic tests for scripts/mw/menu_gate.py (MW-P3-I2a).

All fixtures live under tmp_path — no network, no real project tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# load module under test (path-stable; no package install required)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
MENU_GATE_PATH = REPO_ROOT / "scripts" / "mw" / "menu_gate.py"

_spec = importlib.util.spec_from_file_location("mw_menu_gate", MENU_GATE_PATH)
assert _spec and _spec.loader
menu_gate = importlib.util.module_from_spec(_spec)
sys.modules["mw_menu_gate"] = menu_gate
_spec.loader.exec_module(menu_gate)


# ---------------------------------------------------------------------------
# fixture checklist covering ALL 5 verify types
# ---------------------------------------------------------------------------

FULL_CHECKLIST = textwrap.dedent(
    """\
    meta:
      version: "1.0-test"
      paths:
        harvest: "TOR_Projects/<SITE>/harvest/<menu>/"
        brief: ".work/menu-briefs/<menu>.md"
      menus:
        - alpha
        - beta
    sections:
      - key: "M1_capture"
        title: "Capture"
        items:
          - id: "M1.1"
            check: "old screenshots exist"
            verify: file_glob
            blocking: true
            scope: menu
            glob: "TOR_Projects/<SITE>/harvest/<menu>/old-*.png"
          - id: "M1.2"
            check: "evidence attached"
            verify: evidence_file
            blocking: true
            scope: menu
            glob: "TOR_Projects/<SITE>/harvest/<menu>/evidence-*.png"
      - key: "M2_brief"
        title: "Brief"
        items:
          - id: "M2.1"
            check: "brief has target and owner"
            verify: file_grep
            blocking: true
            scope: menu
            file: ".work/menu-briefs/<menu>.md"
            patterns:
              - "## Target"
              - "owner-confirmed"
          - id: "M2.2"
            check: "queue row with pass marker"
            verify: row_in
            blocking: true
            scope: menu
            file: ".work/menu-queue.md"
            key: "<menu>"
            pass_marker: "✅"
          - id: "M2.3"
            check: "optional note file"
            verify: file_glob
            blocking: false
            scope: menu
            glob: "TOR_Projects/<SITE>/harvest/<menu>/optional-*.txt"
      - key: "M3_site"
        title: "Site level"
        items:
          - id: "M3.1"
            check: "site readme present"
            verify: file_glob
            blocking: true
            scope: site
            glob: "TOR_Projects/<SITE>/README.md"
      - key: "M4_cmd"
        title: "Commands"
        items:
          - id: "M4.1"
            check: "true command"
            verify: command
            blocking: true
            scope: menu
            cmd: "true"
    """
)

# expected structure after parse (for mini-yaml identity check)
EXPECTED_CHECKLIST_KEYS = {
    "meta",
    "sections",
}


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _seed_menu(
    root: Path,
    site: str,
    menu: str,
    *,
    old_png: bool = True,
    evidence: bool = True,
    brief: bool = True,
    brief_patterns: bool = True,
    queue_row: bool = True,
    queue_marker: bool = True,
    optional: bool = True,
    site_readme: bool = True,
) -> None:
    harvest = root / "TOR_Projects" / site / "harvest" / menu
    harvest.mkdir(parents=True, exist_ok=True)
    if old_png:
        (harvest / "old-home.png").write_bytes(b"\x89PNG\r\n")
    if evidence:
        (harvest / "evidence-shot.png").write_bytes(b"\x89PNG\r\n")
    if optional:
        (harvest / "optional-note.txt").write_text("ok\n", encoding="utf-8")

    if brief:
        body = ""
        if brief_patterns:
            body = "## Target\n\nsome target\n\nowner-confirmed\n"
        else:
            body = "## Target\n\nsome target\n"
        _write(root / ".work" / "menu-briefs" / f"{menu}.md", body)

    queue = root / ".work" / "menu-queue.md"
    if queue_row:
        marker = "✅" if queue_marker else "❌"
        line = f"| {menu} | owner | project | 2026-01-01 | 2026-01-02 | {marker} |\n"
        if queue.exists():
            text = queue.read_text(encoding="utf-8")
            if menu not in text:
                queue.write_text(text + line, encoding="utf-8")
        else:
            _write(
                queue,
                "| menu_id | owner | project | acquired | expires | status |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                + line,
            )

    if site_readme:
        _write(root / "TOR_Projects" / site / "README.md", f"# {site}\n")


def _checklist(root: Path, text: str = FULL_CHECKLIST) -> Path:
    return _write(root / ".work" / "menu-checklist.yaml", text)


def _run(
    *args: str,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    # keep hermetic — no accidental HERMES secrets needed
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(MENU_GATE_PATH), *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=full_env,
    )


def _run_json(
    *args: str, cwd: Optional[Path] = None
) -> Tuple[int, Dict[str, Any]]:
    proc = _run(*args, "--json", cwd=cwd)
    data: Dict[str, Any] = {}
    if proc.stdout.strip():
        data = json.loads(proc.stdout)
    return proc.returncode, data


# ===========================================================================
# T1 all_pass
# ===========================================================================

def test_t1_all_pass(tmp_path: Path) -> None:
    site, menu = "SiteA", "alpha"
    _checklist(tmp_path)
    _seed_menu(tmp_path, site, menu)

    code, data = _run_json(site, menu, "--root", str(tmp_path))
    assert code == 0, (code, data)
    assert data["closeable"] is True
    assert data["site"] == site
    assert data["menu"] == menu
    assert data["total"]["fail"] == 0
    assert data["total"]["pass"] == data["total"]["count"]
    assert data["total"]["count"] > 0
    assert data["blocking_red"] == []
    # live counts: 6 blocking+nonblocking items (M1.1 M1.2 M2.1 M2.2 M2.3 M3.1 M4.1) = 7
    assert data["total"]["count"] == 7
    assert data["total"]["pass"] == 7


# ===========================================================================
# T2 blocking_red
# ===========================================================================

def test_t2_blocking_red(tmp_path: Path) -> None:
    site, menu = "SiteA", "alpha"
    _checklist(tmp_path)
    _seed_menu(tmp_path, site, menu, old_png=False)  # M1.1 missing

    code, data = _run_json(site, menu, "--root", str(tmp_path))
    assert code == 1
    assert data["closeable"] is False
    assert "M1.1" in data["blocking_red"]
    assert data["total"]["fail"] >= 1
    assert data["total"]["pass"] + data["total"]["fail"] == data["total"]["count"]


# ===========================================================================
# T3 non_blocking_red_still_closeable
# ===========================================================================

def test_t3_non_blocking_red_still_closeable(tmp_path: Path) -> None:
    site, menu = "SiteA", "alpha"
    _checklist(tmp_path)
    _seed_menu(tmp_path, site, menu, optional=False)  # M2.3 non-blocking

    code, data = _run_json(site, menu, "--root", str(tmp_path))
    assert code == 0
    assert data["closeable"] is True
    assert "M2.3" not in data["blocking_red"]
    # fail count includes non-blocking
    assert data["total"]["fail"] == 1
    assert data["total"]["pass"] == data["total"]["count"] - 1


# ===========================================================================
# T4 each verify type independently
# ===========================================================================

def test_t4_file_glob_hit_miss(tmp_path: Path) -> None:
    root = tmp_path
    (root / "a").mkdir()
    (root / "a" / "hit.png").write_bytes(b"x")
    assert menu_gate.eval_file_glob(root, "a/hit.png").status == "pass"
    assert menu_gate.eval_file_glob(root, "a/*.png").status == "pass"
    assert menu_gate.eval_file_glob(root, "a/missing-*.png").status == "fail"
    assert menu_gate.eval_file_glob(root, "nope/**/*.png").status == "fail"


def test_t4_evidence_file_present_absent(tmp_path: Path) -> None:
    root = tmp_path
    (root / "e").mkdir()
    (root / "e" / "evidence-1.png").write_bytes(b"x")
    assert menu_gate.eval_evidence_file(root, "e/evidence-*.png").status == "pass"
    assert menu_gate.eval_evidence_file(root, "e/nope-*.png").status == "fail"


def test_t4_file_grep_all_present_vs_missing(tmp_path: Path) -> None:
    root = tmp_path
    f = root / "brief.md"
    f.write_text("## Target\nowner-confirmed\n", encoding="utf-8")
    ok = menu_gate.eval_file_grep(root, "brief.md", ["## Target", "owner-confirmed"])
    assert ok.status == "pass"
    miss = menu_gate.eval_file_grep(root, "brief.md", ["## Target", "MISSING"])
    assert miss.status == "fail"
    assert "MISSING" in miss.reason
    nofile = menu_gate.eval_file_grep(root, "gone.md", ["x"])
    assert nofile.status == "fail"


def test_t4_row_in_present_marker_vs_missing(tmp_path: Path) -> None:
    root = tmp_path
    q = root / "queue.md"
    q.write_text("| alpha | owner | ✅ |\n| beta | owner | ❌ |\n", encoding="utf-8")
    assert menu_gate.eval_row_in(root, "queue.md", "alpha", "✅").status == "pass"
    assert menu_gate.eval_row_in(root, "queue.md", "beta", "✅").status == "fail"
    assert menu_gate.eval_row_in(root, "queue.md", "gamma", "✅").status == "fail"
    assert menu_gate.eval_row_in(root, "queue.md", "beta", None).status == "pass"


def test_t4_command_exit0_vs_exit1(tmp_path: Path) -> None:
    ok, detail = menu_gate.eval_command("true", timeout=10, cwd=tmp_path)
    assert ok.status == "pass"
    assert detail["exit_code"] == 0

    bad, detail2 = menu_gate.eval_command("false", timeout=10, cwd=tmp_path)
    assert bad.status == "fail"
    assert detail2["exit_code"] != 0


# ===========================================================================
# T5 templating
# ===========================================================================

def test_t5_templating_resolves_paths(tmp_path: Path) -> None:
    site, menu = "AcmeCorp", "nav-main"
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "t5"
            sections:
              - key: "T"
                title: "T"
                items:
                  - id: "T.1"
                    check: "templated glob"
                    verify: file_glob
                    blocking: true
                    glob: "data/<SITE>/<menu>/ok.txt"
                  - id: "T.2"
                    check: "templated grep"
                    verify: file_grep
                    blocking: true
                    file: "data/<site>/<menu>/note.md"
                    patterns:
                      - "MENU=<menu>"
                      - "SITE=<SITE>"
            """
        ),
    )
    target = tmp_path / "data" / site / menu
    target.mkdir(parents=True)
    (target / "ok.txt").write_text("yes\n", encoding="utf-8")
    (target / "note.md").write_text(
        f"MENU={menu}\nSITE={site}\n", encoding="utf-8"
    )

    code, data = _run_json(site, menu, "--root", str(tmp_path))
    assert code == 0
    assert data["closeable"] is True
    assert data["total"]["pass"] == 2

    # also unit-level apply_template
    assert (
        menu_gate.apply_template("TOR/<SITE>/harvest/<menu>/x", site, menu)
        == f"TOR/{site}/harvest/{menu}/x"
    )
    assert (
        menu_gate.apply_template("<site>/<menu>", site, menu)
        == f"{site}/{menu}"
    )


# ===========================================================================
# T6 command_timeout
# ===========================================================================

def test_t6_command_timeout(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "t6"
            sections:
              - key: "TO"
                title: "timeout"
                items:
                  - id: "TO.1"
                    check: "sleep exceeds timeout"
                    verify: command
                    blocking: true
                    cmd: "sleep 5"
            """
        ),
    )
    code, data = _run_json(
        "S", "m", "--root", str(tmp_path), "--cmd-timeout", "1"
    )
    assert code == 1
    assert data["closeable"] is False
    assert "TO.1" in data["blocking_red"]
    # reason should mention timeout
    reasons = []
    for sec in data["sections"]:
        for it in sec["items"]:
            if it["id"] == "TO.1":
                reasons.append(it["reason"])
    assert reasons
    assert "timed out" in reasons[0].lower() or "timeout" in reasons[0].lower()


# ===========================================================================
# T7 missing checklist / unknown verify
# ===========================================================================

def test_t7_missing_checklist(tmp_path: Path) -> None:
    proc = _run("SiteX", "menu1", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "checklist" in (proc.stderr + proc.stdout).lower()


def test_t7_unknown_verify_type(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "t7"
            sections:
              - key: "U"
                title: "unknown"
                items:
                  - id: "U.1"
                    check: "bad type"
                    verify: not_a_real_type
                    blocking: true
            """
        ),
    )
    proc = _run("S", "m", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "unknown" in (proc.stderr + proc.stdout).lower()


# ===========================================================================
# T8 json_shape
# ===========================================================================

def test_t8_json_shape(tmp_path: Path) -> None:
    site, menu = "SiteA", "alpha"
    _checklist(tmp_path)
    _seed_menu(tmp_path, site, menu)

    code, data = _run_json(site, menu, "--root", str(tmp_path))
    assert code == 0
    for key in ("site", "menu", "total", "blocking_red", "sections", "closeable"):
        assert key in data, f"missing key {key}"
    for key in ("pass", "fail", "count"):
        assert key in data["total"]
    assert isinstance(data["blocking_red"], list)
    assert isinstance(data["sections"], list)
    assert isinstance(data["closeable"], bool)
    # live counts consistency
    assert data["total"]["pass"] + data["total"]["fail"] == data["total"]["count"]
    # section item shape
    for sec in data["sections"]:
        assert "key" in sec
        assert "pass" in sec
        assert "count" in sec
        assert "items" in sec
        for it in sec["items"]:
            assert it["status"] in ("pass", "fail")
            assert "id" in it
            assert "verify" in it
            assert "reason" in it


# ===========================================================================
# T9 minimal_yaml_loader
# ===========================================================================

def test_t9_minimal_yaml_loader_parses_fixture(tmp_path: Path) -> None:
    """Force fallback loader even if PyYAML is installed."""
    path = _write(tmp_path / "cl.yaml", FULL_CHECKLIST)

    # force mini path via module flag
    old = menu_gate._FORCE_MINI_YAML
    try:
        menu_gate._FORCE_MINI_YAML = True
        data = menu_gate.load_yaml_file(path, force_mini=True)
    finally:
        menu_gate._FORCE_MINI_YAML = old

    assert isinstance(data, dict)
    assert "meta" in data
    assert data["meta"]["version"] == "1.0-test"
    assert data["meta"]["paths"]["harvest"] == "TOR_Projects/<SITE>/harvest/<menu>/"
    assert data["meta"]["menus"] == ["alpha", "beta"]

    sections = data["sections"]
    assert isinstance(sections, list)
    assert len(sections) >= 4
    assert sections[0]["key"] == "M1_capture"
    items = sections[0]["items"]
    assert items[0]["id"] == "M1.1"
    assert items[0]["verify"] == "file_glob"
    assert items[0]["blocking"] is True
    assert items[0]["glob"] == "TOR_Projects/<SITE>/harvest/<menu>/old-*.png"

    # file_grep patterns list
    m2 = next(s for s in sections if s["key"] == "M2_brief")
    m21 = next(i for i in m2["items"] if i["id"] == "M2.1")
    assert m21["patterns"] == ["## Target", "owner-confirmed"]

    # non-blocking false
    m23 = next(i for i in m2["items"] if i["id"] == "M2.3")
    assert m23["blocking"] is False

    # site scope
    m3 = next(s for s in sections if s["key"] == "M3_site")
    assert m3["items"][0]["scope"] == "site"

    # command
    m4 = next(s for s in sections if s["key"] == "M4_cmd")
    assert m4["items"][0]["cmd"] == "true"

    # if PyYAML available, structure should match
    if menu_gate._yaml is not None:
        py = menu_gate.load_yaml_text(FULL_CHECKLIST, force_mini=False)
        mini = menu_gate.load_yaml_text(FULL_CHECKLIST, force_mini=True)
        assert py["meta"]["version"] == mini["meta"]["version"]
        assert len(py["sections"]) == len(mini["sections"])
        assert py["sections"][0]["key"] == mini["sections"][0]["key"]
        assert py["sections"][0]["items"][0]["id"] == mini["sections"][0]["items"][0]["id"]
        # patterns
        py_m21 = next(
            i
            for s in py["sections"]
            if s["key"] == "M2_brief"
            for i in s["items"]
            if i["id"] == "M2.1"
        )
        mini_m21 = next(
            i
            for s in mini["sections"]
            if s["key"] == "M2_brief"
            for i in s["items"]
            if i["id"] == "M2.1"
        )
        assert py_m21["patterns"] == mini_m21["patterns"]


def test_t9_mini_yaml_inline_list_and_booleans() -> None:
    text = textwrap.dedent(
        """\
        # comment
        meta:
          version: "x"
          flag: true
          off: false
        sections:
          - key: "A"
            title: "A"
            items:
              - id: "A.1"
                verify: file_grep
                blocking: false
                patterns: ["one", "two"]
        """
    )
    data = menu_gate._mini_yaml(text)
    assert data["meta"]["flag"] is True
    assert data["meta"]["off"] is False
    assert data["sections"][0]["items"][0]["patterns"] == ["one", "two"]
    assert data["sections"][0]["items"][0]["blocking"] is False


# ===========================================================================
# T10 all_mode
# ===========================================================================

def test_t10_all_mode_one_closeable_one_not(tmp_path: Path) -> None:
    site = "SiteA"
    _checklist(tmp_path)
    # alpha fully seeded
    _seed_menu(tmp_path, site, "alpha")
    # beta missing blocking evidence (old png)
    _seed_menu(tmp_path, site, "beta", old_png=False)

    code, data = _run_json(site, "--all", "--root", str(tmp_path))
    assert code == 1
    assert data["menu"] == "--all"
    assert data["closeable"] is False
    assert "menus" in data
    assert len(data["menus"]) == 2

    by_menu = {m["menu"]: m for m in data["menus"]}
    assert by_menu["alpha"]["closeable"] is True
    assert by_menu["beta"]["closeable"] is False
    assert "M1.1" in by_menu["beta"]["blocking_red"]

    # site-scope counted once in aggregate: M3.1 appears once in site_scope
    assert "site_scope" in data
    site_ids = [
        it["id"]
        for sec in data["site_scope"]["sections"]
        for it in sec["items"]
    ]
    assert site_ids.count("M3.1") == 1

    # aggregate total = site items + sum of menu-only items (no double site)
    # 1 site (M3.1) + 2 menus * 6 menu-scope items = 1 + 12 = 13
    assert data["total"]["count"] == 13
    assert data["total"]["pass"] + data["total"]["fail"] == data["total"]["count"]


def test_t10_all_both_pass(tmp_path: Path) -> None:
    site = "SiteA"
    _checklist(tmp_path)
    _seed_menu(tmp_path, site, "alpha")
    _seed_menu(tmp_path, site, "beta")

    code, data = _run_json(site, "--all", "--root", str(tmp_path))
    assert code == 0
    assert data["closeable"] is True
    assert data["total"]["fail"] == 0


# ===========================================================================
# extra: human mode + discover menus from harvest dirs
# ===========================================================================

def test_human_mode_output_mentions_closeable(tmp_path: Path) -> None:
    site, menu = "SiteA", "alpha"
    _checklist(tmp_path)
    _seed_menu(tmp_path, site, menu)
    proc = _run(site, menu, "--root", str(tmp_path))
    assert proc.returncode == 0
    assert "CLOSEABLE=YES" in proc.stdout
    assert "M1_capture:" in proc.stdout


def test_discover_menus_from_harvest_when_no_meta_list(tmp_path: Path) -> None:
    site = "SiteB"
    cl = textwrap.dedent(
        """\
        meta:
          version: "d"
          paths:
            harvest: "TOR_Projects/<SITE>/harvest/<menu>/"
        sections:
          - key: "X"
            title: "X"
            items:
              - id: "X.1"
                check: "readme"
                verify: file_glob
                blocking: true
                scope: site
                glob: "TOR_Projects/<SITE>/README.md"
        """
    )
    _write(tmp_path / ".work" / "menu-checklist.yaml", cl)
    for m in ("m1", "m2"):
        (tmp_path / "TOR_Projects" / site / "harvest" / m).mkdir(parents=True)
    _write(tmp_path / "TOR_Projects" / site / "README.md", "# ok\n")

    code, data = _run_json(site, "--all", "--root", str(tmp_path))
    assert code == 0
    names = sorted(m["menu"] for m in data["menus"])
    assert names == ["m1", "m2"]


def test_explicit_checklist_path(tmp_path: Path) -> None:
    cl = tmp_path / "custom.yaml"
    _write(
        cl,
        textwrap.dedent(
            """\
            meta:
              version: "c"
            sections:
              - key: "C"
                title: "C"
                items:
                  - id: "C.1"
                    verify: file_glob
                    blocking: true
                    glob: "marker.txt"
            """
        ),
    )
    (tmp_path / "marker.txt").write_text("x\n", encoding="utf-8")
    code, data = _run_json(
        "S", "m", "--root", str(tmp_path), "--checklist", str(cl)
    )
    assert code == 0
    assert data["closeable"] is True


def test_legacy_top_level_sections_shape(tmp_path: Path) -> None:
    """Legacy: section keys as top-level mappings (not under sections:)."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "legacy"
            M1_legacy:
              title: "Legacy"
              items:
                - id: "L.1"
                  verify: file_glob
                  blocking: true
                  glob: "ok.txt"
            """
        ),
    )
    (tmp_path / "ok.txt").write_text("1\n", encoding="utf-8")
    code, data = _run_json("S", "m", "--root", str(tmp_path))
    assert code == 0
    assert data["sections"][0]["key"] == "M1_legacy"


def test_command_uses_list_argv_no_shell(tmp_path: Path) -> None:
    """Ensure shell metacharacters are NOT expanded (shell=False)."""
    # With shell=True this would create shell_out.txt; with list argv it must not.
    result, _detail = menu_gate.eval_command(
        "echo pwned > shell_out.txt", timeout=5, cwd=tmp_path
    )
    assert not (tmp_path / "shell_out.txt").exists()
    # echo still may exit 0 printing the tokens — the critical invariant is no redirect.
    _ = result  # status not the contract here; no shell file is.


def test_file_grep_single_string_pattern(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "s"
            sections:
              - key: "G"
                title: "G"
                items:
                  - id: "G.1"
                    verify: file_grep
                    blocking: true
                    file: "f.md"
                    patterns: "ONLY_ONE"
            """
        ),
    )
    (tmp_path / "f.md").write_text("ONLY_ONE\n", encoding="utf-8")
    code, data = _run_json("S", "m", "--root", str(tmp_path))
    assert code == 0
    assert data["total"]["pass"] == 1


def test_all_mode_no_menus_exit_2(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "empty"
              paths:
                harvest: "gone/<SITE>/harvest/<menu>/"
            sections:
              - key: "Z"
                title: "Z"
                items:
                  - id: "Z.1"
                    verify: file_glob
                    blocking: true
                    glob: "x"
            """
        ),
    )
    proc = _run("S", "--all", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "no menus" in (proc.stderr + proc.stdout).lower()


# ===========================================================================
# FIX A — file_grep missing/empty patterns = config error (exit 2), never green
# ===========================================================================

def test_fix_a_file_grep_missing_patterns_exit_2(tmp_path: Path) -> None:
    """file_grep with no patterns must not vacuous-pass (all of [] is true)."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-a"
            sections:
              - key: "G"
                title: "G"
                items:
                  - id: "G.empty"
                    check: "patterns omitted"
                    verify: file_grep
                    blocking: true
                    file: "f.md"
            """
        ),
    )
    (tmp_path / "f.md").write_text("anything\n", encoding="utf-8")
    proc = _run("S", "m", "--root", str(tmp_path))
    assert proc.returncode == 2
    combined = (proc.stderr + proc.stdout).lower()
    assert "no patterns" in combined
    assert "g.empty" in combined


def test_fix_a_file_grep_empty_patterns_list_exit_2(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-a2"
            sections:
              - key: "G"
                title: "G"
                items:
                  - id: "G.empty_list"
                    verify: file_grep
                    blocking: true
                    file: "f.md"
                    patterns: []
            """
        ),
    )
    (tmp_path / "f.md").write_text("x\n", encoding="utf-8")
    proc = _run("S", "m", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "no patterns" in (proc.stderr + proc.stdout).lower()


# ===========================================================================
# FIX B — mini-yaml dash-on-own-line + structural fail-closed (never 0/0 green)
# ===========================================================================

def test_fix_b_dash_on_own_line_parses_and_evaluates(tmp_path: Path) -> None:
    """Standard YAML block: ``-`` alone, mapping keys on next indented lines."""
    cl_text = textwrap.dedent(
        """\
        meta:
          version: "dash-alone"
        sections:
          - key: "D"
            title: "Dash alone items"
            items:
              -
                id: "D.1"
                check: "marker present"
                verify: file_glob
                blocking: true
                glob: "marker.txt"
        """
    )
    _write(tmp_path / ".work" / "menu-checklist.yaml", cl_text)
    (tmp_path / "marker.txt").write_text("ok\n", encoding="utf-8")

    # mini parser must keep the item
    old = menu_gate._FORCE_MINI_YAML
    try:
        menu_gate._FORCE_MINI_YAML = True
        data = menu_gate.load_yaml_text(cl_text, force_mini=True)
    finally:
        menu_gate._FORCE_MINI_YAML = old

    sections = data["sections"]
    assert sections[0]["items"][0]["id"] == "D.1"
    assert sections[0]["items"][0]["verify"] == "file_glob"

    code, report = _run_json("S", "m", "--root", str(tmp_path))
    assert code == 0
    assert report["closeable"] is True
    assert report["total"]["count"] == 1
    assert report["total"]["pass"] == 1


def test_fix_b_structural_zero_items_never_closeable(tmp_path: Path) -> None:
    """Empty items list → exit 2; never print closeable=True with 0 items."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "zero"
            sections:
              - key: "E"
                title: "Empty"
                items: []
            """
        ),
    )
    proc = _run("S", "m", "--root", str(tmp_path), "--json")
    assert proc.returncode == 2
    out = proc.stdout + proc.stderr
    # must not emit a green closeable report
    if proc.stdout.strip():
        try:
            data = json.loads(proc.stdout)
            assert data.get("closeable") is not True
            assert not (
                data.get("closeable") is True and data.get("total", {}).get("count") == 0
            )
        except json.JSONDecodeError:
            pass
    assert "closeable=true" not in out.lower() or "error" in out.lower()
    # stronger: exit 2 means config error, not green
    assert proc.returncode == 2


def test_fix_b_missing_id_or_verify_exit_2(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "noid"
            sections:
              - key: "N"
                title: "N"
                items:
                  - check: "no id field"
                    verify: file_glob
                    blocking: true
                    glob: "x"
            """
        ),
    )
    proc = _run("S", "m", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "missing id" in (proc.stderr + proc.stdout).lower()


def test_fix_b_mini_yaml_ambiguous_raises_exit_2(tmp_path: Path) -> None:
    """Forced mini path + unrepresentable construct → exit 2, never 0/0 green."""
    # Mapping value line without ':' is unrepresentable → MiniYamlError
    bad = textwrap.dedent(
        """\
        meta:
          version: "bad"
        sections:
          - key: "X"
            title: "X"
            items:
              - id: "I.1"
                verify: file_glob
                blocking: true
                glob: "x.txt"
                this_line_has_no_colon
        """
    )
    _write(tmp_path / ".work" / "menu-checklist.yaml", bad)

    with pytest.raises(menu_gate.MiniYamlError):
        menu_gate.load_yaml_text(bad, force_mini=True)

    # In-process CLI with force_mini → exit 2 (never closeable with 0 items)
    old = menu_gate._FORCE_MINI_YAML
    try:
        menu_gate._FORCE_MINI_YAML = True
        code = menu_gate.run(["S", "m", "--root", str(tmp_path)])
    finally:
        menu_gate._FORCE_MINI_YAML = old
    assert code == 2


# ===========================================================================
# FIX C — path escapes --root must fail closed (red), never green from outside
# ===========================================================================

def test_fix_c_path_escape_file_grep_red(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET_PASS_MARKER\n", encoding="utf-8")

    _write(
        root / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "escape"
            sections:
              - key: "E"
                title: "Escape"
                items:
                  - id: "E.1"
                    check: "must not read outside root"
                    verify: file_grep
                    blocking: true
                    file: "../outside.txt"
                    patterns:
                      - "SECRET_PASS_MARKER"
            """
        ),
    )
    code, data = _run_json("S", "m", "--root", str(root))
    assert code == 1
    assert data["closeable"] is False
    assert "E.1" in data["blocking_red"]
    reasons = [
        it["reason"]
        for sec in data["sections"]
        for it in sec["items"]
        if it["id"] == "E.1"
    ]
    assert reasons
    assert "path escapes --root" in reasons[0]


def test_fix_c_path_escape_file_glob_red(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x\n", encoding="utf-8")

    _write(
        root / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "escape-glob"
            sections:
              - key: "E"
                title: "Escape"
                items:
                  - id: "E.g"
                    verify: file_glob
                    blocking: true
                    glob: "../outside.txt"
            """
        ),
    )
    code, data = _run_json("S", "m", "--root", str(root))
    assert code == 1
    assert data["closeable"] is False
    assert "E.g" in data["blocking_red"]
    reasons = [
        it["reason"]
        for sec in data["sections"]
        for it in sec["items"]
        if it["id"] == "E.g"
    ]
    assert any("path escapes --root" in r for r in reasons)


# ===========================================================================
# FIX D — site/menu id injection via spaces / option-like tokens → exit 2
# ===========================================================================

def test_fix_d_menu_with_space_exit_2(tmp_path: Path) -> None:
    _checklist(tmp_path)
    _seed_menu(tmp_path, "SiteA", "alpha")
    proc = _run("SiteA", "safe extra", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "invalid site/menu id" in (proc.stderr + proc.stdout).lower()


def test_fix_d_menu_option_like_exit_2(tmp_path: Path) -> None:
    _checklist(tmp_path)
    proc = _run("SiteA", "-rf", "--root", str(tmp_path))
    assert proc.returncode == 2
    assert "invalid site/menu id" in (proc.stderr + proc.stdout).lower()


def test_fix_d_discovered_menu_with_space_skipped(tmp_path: Path) -> None:
    """FIX D2: skipped invalid discovered menu → exit 2, never certify closeable."""
    site = "SiteB"
    cl = textwrap.dedent(
        """\
        meta:
          version: "d-space"
          paths:
            harvest: "TOR_Projects/<SITE>/harvest/<menu>/"
        sections:
          - key: "X"
            title: "X"
            items:
              - id: "X.1"
                check: "readme"
                verify: file_glob
                blocking: true
                scope: site
                glob: "TOR_Projects/<SITE>/README.md"
        """
    )
    _write(tmp_path / ".work" / "menu-checklist.yaml", cl)
    # valid + invalid discovered dirs
    (tmp_path / "TOR_Projects" / site / "harvest" / "goodmenu").mkdir(parents=True)
    (tmp_path / "TOR_Projects" / site / "harvest" / "bad menu").mkdir(parents=True)
    _write(tmp_path / "TOR_Projects" / site / "README.md", "# ok\n")

    proc = _run(site, "--all", "--root", str(tmp_path), "--json")
    # incomplete coverage when any discovered menu is skipped → exit 2
    combined = (proc.stderr + proc.stdout).lower()
    assert "bad menu" in combined or "skip" in combined
    assert proc.returncode == 2
    assert "cannot certify" in combined or "skipped" in combined
    # must not print closeable=true (JSON report is not a green cert)
    if proc.stdout.strip():
        try:
            data = json.loads(proc.stdout)
            assert data.get("closeable") is not True
        except json.JSONDecodeError:
            pass


# ===========================================================================
# FIX E — row_in must match real markdown table rows only
# ===========================================================================

def test_fix_e_row_in_ignores_comment_and_heading(tmp_path: Path) -> None:
    root = tmp_path
    q = root / "queue.md"
    q.write_text(
        "<!-- menuX ✅ -->\n"
        "# menuX ✅\n"
        "| menu_id | status |\n"
        "| --- | --- |\n"
        "| other | ❌ |\n",
        encoding="utf-8",
    )
    # comment + heading must NOT pass
    assert menu_gate.eval_row_in(root, "queue.md", "menuX", "✅").status == "fail"

    # real table row passes
    q.write_text(
        "<!-- menuX ✅ -->\n"
        "# menuX ✅\n"
        "| menu_id | status |\n"
        "| --- | --- |\n"
        "| menuX | ✅ |\n",
        encoding="utf-8",
    )
    assert menu_gate.eval_row_in(root, "queue.md", "menuX", "✅").status == "pass"


def test_fix_e_row_in_checklist_comment_not_closeable(tmp_path: Path) -> None:
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "row-e"
            sections:
              - key: "R"
                title: "R"
                items:
                  - id: "R.1"
                    verify: row_in
                    blocking: true
                    file: "q.md"
                    key: "menuX"
                    pass_marker: "✅"
            """
        ),
    )
    (tmp_path / "q.md").write_text(
        "<!-- menuX ✅ -->\n# menuX ✅\n",
        encoding="utf-8",
    )
    code, data = _run_json("S", "m", "--root", str(tmp_path))
    assert code == 1
    assert data["closeable"] is False
    assert "R.1" in data["blocking_red"]


# ===========================================================================
# FIX A2 — empty/whitespace pattern member vacuous-matches every file → exit 2
# ===========================================================================

def test_fix_a2_empty_pattern_member_exit_2(tmp_path: Path) -> None:
    """patterns: [\"\"] is a substring of everything → config error, not green."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-a2-empty"
            sections:
              - key: "G"
                title: "G"
                items:
                  - id: "G.empty_member"
                    verify: file_grep
                    blocking: true
                    file: "f.md"
                    patterns:
                      - ""
            """
        ),
    )
    (tmp_path / "f.md").write_text("anything\n", encoding="utf-8")
    proc = _run("S", "m", "--root", str(tmp_path))
    assert proc.returncode == 2
    combined = (proc.stderr + proc.stdout).lower()
    assert "empty pattern" in combined
    assert "g.empty_member" in combined
    assert "closeable=true" not in combined


def test_fix_a2_whitespace_pattern_member_exit_2(tmp_path: Path) -> None:
    """patterns: [\"  \"] after strip is empty → exit 2."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-a2-ws"
            sections:
              - key: "G"
                title: "G"
                items:
                  - id: "G.ws_member"
                    verify: file_grep
                    blocking: true
                    file: "f.md"
                    patterns:
                      - "  "
            """
        ),
    )
    (tmp_path / "f.md").write_text("anything\n", encoding="utf-8")
    proc = _run("S", "m", "--root", str(tmp_path))
    assert proc.returncode == 2
    combined = (proc.stderr + proc.stdout).lower()
    assert "empty pattern" in combined
    assert "g.ws_member" in combined


# ===========================================================================
# FIX B2 — malformed section/item must not be dropped while survivors go green
# ===========================================================================

def test_fix_b2_malformed_section_item_string_exit_2(tmp_path: Path) -> None:
    """One good section + one section whose item is a bare string → exit 2."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-b2-str"
            sections:
              - key: "GOOD"
                title: "Good"
                items:
                  - id: "G.1"
                    verify: file_glob
                    blocking: true
                    glob: "marker.txt"
              - key: "BAD"
                title: "Bad"
                items:
                  - "this-is-a-bare-string-not-a-mapping"
            """
        ),
    )
    (tmp_path / "marker.txt").write_text("ok\n", encoding="utf-8")
    proc = _run("S", "m", "--root", str(tmp_path), "--json")
    assert proc.returncode == 2
    out = (proc.stderr + proc.stdout).lower()
    assert "not a mapping" in out or "must be a mapping" in out
    assert "closeable=true" not in out
    if proc.stdout.strip():
        try:
            data = json.loads(proc.stdout)
            assert data.get("closeable") is not True
        except json.JSONDecodeError:
            pass


def test_fix_b2_malformed_section_items_scalar_exit_2(tmp_path: Path) -> None:
    """items is a scalar (not a list) → exit 2; survivors must not green alone."""
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-b2-scalar"
            sections:
              - key: "GOOD"
                title: "Good"
                items:
                  - id: "G.1"
                    verify: file_glob
                    blocking: true
                    glob: "marker.txt"
              - key: "BAD"
                title: "Bad"
                items: "not-a-list"
            """
        ),
    )
    (tmp_path / "marker.txt").write_text("ok\n", encoding="utf-8")
    proc = _run("S", "m", "--root", str(tmp_path), "--json")
    assert proc.returncode == 2
    out = (proc.stderr + proc.stdout).lower()
    assert "items must be a non-empty list" in out or "not a mapping" in out
    assert "closeable=true" not in out


# ===========================================================================
# FIX D2 — --all + invalid discovered menu must not exit 0 closeable
# ===========================================================================

def test_fix_d2_all_mode_skips_invalid_menu_exit_2(tmp_path: Path) -> None:
    """Harvest has valid green menu AND dir 'bad menu' → run does NOT exit 0."""
    site = "SiteA"
    cl = textwrap.dedent(
        """\
        meta:
          version: "fix-d2"
          paths:
            harvest: "TOR_Projects/<SITE>/harvest/<menu>/"
        sections:
          - key: "X"
            title: "X"
            items:
              - id: "X.1"
                check: "marker"
                verify: file_glob
                blocking: true
                scope: menu
                glob: "TOR_Projects/<SITE>/harvest/<menu>/ok.txt"
        """
    )
    _write(tmp_path / ".work" / "menu-checklist.yaml", cl)
    good = tmp_path / "TOR_Projects" / site / "harvest" / "goodmenu"
    good.mkdir(parents=True)
    (good / "ok.txt").write_text("yes\n", encoding="utf-8")
    (tmp_path / "TOR_Projects" / site / "harvest" / "bad menu").mkdir(parents=True)

    proc = _run(site, "--all", "--root", str(tmp_path), "--json")
    assert proc.returncode != 0
    assert proc.returncode == 2
    combined = (proc.stderr + proc.stdout).lower()
    assert "bad menu" in combined
    assert "cannot certify" in combined or "skipped" in combined
    assert "closeable=true" not in combined


# ===========================================================================
# FIX E2 — row_in key must match an EXACT table cell (not substring)
# ===========================================================================

def test_fix_e2_row_in_exact_cell_not_substring(tmp_path: Path) -> None:
    """key 'menuX' must NOT pass row whose only relevant cell is 'menuX-old'."""
    root = tmp_path
    q = root / "queue.md"
    # adversarial: substring match would false-green on menuX-old
    q.write_text(
        "| menu_id | status |\n"
        "| --- | --- |\n"
        "| menuX-old | ✅ |\n",
        encoding="utf-8",
    )
    assert menu_gate.eval_row_in(root, "queue.md", "menuX", "✅").status == "fail"

    # exact cell + marker must pass
    q.write_text(
        "| menu_id | status |\n"
        "| --- | --- |\n"
        "| menuX | ✅ |\n",
        encoding="utf-8",
    )
    assert menu_gate.eval_row_in(root, "queue.md", "menuX", "✅").status == "pass"

    # checklist path: substring-only evidence must not close
    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-e2"
            sections:
              - key: "R"
                title: "R"
                items:
                  - id: "R.e2"
                    verify: row_in
                    blocking: true
                    file: "queue.md"
                    key: "menuX"
                    pass_marker: "✅"
            """
        ),
    )
    q.write_text(
        "| menu_id | status |\n"
        "| --- | --- |\n"
        "| menuX-old | ✅ |\n",
        encoding="utf-8",
    )
    code, data = _run_json("S", "m", "--root", str(tmp_path))
    assert code == 1
    assert data["closeable"] is False
    assert "R.e2" in data["blocking_red"]


# ===========================================================================
# FIX F — file_glob / evidence_file must not treat a DIRECTORY as evidence
# ===========================================================================

def test_fix_f_glob_matching_directory_is_red(tmp_path: Path) -> None:
    """A directory matching the glob is NOT evidence; real file is."""
    root = tmp_path
    # directory named like a file target
    (root / "proof.png").mkdir()
    assert menu_gate.eval_file_glob(root, "proof.png").status == "fail"
    assert menu_gate.eval_evidence_file(root, "proof.png").status == "fail"
    assert menu_gate.eval_file_glob(root, "proof.*").status == "fail"

    # real file → pass
    (root / "real.png").write_bytes(b"\x89PNG")
    assert menu_gate.eval_file_glob(root, "real.png").status == "pass"
    assert menu_gate.eval_evidence_file(root, "real.png").status == "pass"

    _write(
        tmp_path / ".work" / "menu-checklist.yaml",
        textwrap.dedent(
            """\
            meta:
              version: "fix-f"
            sections:
              - key: "F"
                title: "F"
                items:
                  - id: "F.dir"
                    verify: file_glob
                    blocking: true
                    glob: "proof.png"
            """
        ),
    )
    code, data = _run_json("S", "m", "--root", str(tmp_path))
    assert code == 1
    assert data["closeable"] is False
    assert "F.dir" in data["blocking_red"]

"""Tests that ``scripts/check-windows-footguns.py`` sees aliased module imports.

Every member-naming rule in the linter is anchored with a leading ``\\b``
(``\\bsignal\\.SIGKILL\\b``, ``\\bos\\.setsid\\b``, ...). ``_`` is a word
character, so there is no word boundary between ``_`` and ``s`` in
``_signal.SIGKILL`` and the rule never fires. ``import signal as _signal`` --
which thirteen modules in this repo use -- therefore made a file invisible to
those rules, and the linter reported a clean tree.

These tests pin the two halves: the alias map is built from the import lines
(including function-local ones), and ``scan_file`` flags an aliased footgun
exactly as it flags the canonical spelling.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _load_linter_module():
    """Import the linter script as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _scan(linter, tmp_path: Path, source: str) -> list[tuple[int, str, str]]:
    """Run the real ``scan_file`` over *source*; return (lineno, line, rule name)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "subject.py"
    path.write_text(source, encoding="utf-8")
    return [(n, line, fg.name) for n, line, fg in linter.scan_file(path, linter.FOOTGUNS)]


# --- the alias map ---------------------------------------------------------


def test_module_level_alias_is_mapped(linter):
    assert linter.module_aliases("import signal as _signal\n") == {"_signal": "signal"}


def test_function_local_alias_is_mapped(linter):
    """Most of this repo's aliased imports sit inside a def."""
    source = "def reap():\n    import signal as _signal\n    return _signal.SIGKILL\n"

    assert linter.module_aliases(source) == {"_signal": "signal"}


def test_several_aliases_are_mapped(linter):
    source = "import os as _os\nimport signal as _sig\nimport subprocess as _sp\n"

    assert linter.module_aliases(source) == {
        "_os": "os",
        "_sig": "signal",
        "_sp": "subprocess",
    }


def test_unrelated_module_alias_is_ignored(linter):
    assert linter.module_aliases("import time as _time\nimport json as j\n") == {}


def test_alias_shadowing_another_canonical_name_is_skipped(linter):
    """`import signal as os` would make every os rule lie; refuse to guess."""
    assert linter.module_aliases("import signal as os\n") == {}


def test_redundant_self_alias_is_not_recorded(linter):
    assert linter.module_aliases("import os as os\n") == {}


def test_normalizer_is_none_when_nothing_to_rewrite(linter):
    assert linter.alias_normalizer({}) is None


def test_normalizer_does_not_rewrite_a_longer_identifier(linter):
    """`my_signal.` must survive: the alias is `_signal`, not a substring match."""
    normalize = linter.alias_normalizer({"_signal": "signal"})

    assert normalize("my_signal.SIGKILL") == "my_signal.SIGKILL"
    assert normalize("_signal.SIGKILL") == "signal.SIGKILL"


# --- end-to-end through scan_file -----------------------------------------


ALIASED = """\
import os as _os
import signal as _signal


def reap(pid):
    _os.kill(pid, _signal.SIGKILL)
    _os.killpg(_os.getpgid(pid), _signal.SIGKILL)


def whoami():
    return _os.getuid()


def detach():
    _os.setsid()
"""

CANONICAL = ALIASED.replace("import os as _os", "import os").replace(
    "import signal as _signal", "import signal"
).replace("_os.", "os.").replace("_signal.", "signal.")


def test_aliased_footguns_are_flagged(linter, tmp_path):
    hits = _scan(linter, tmp_path, ALIASED)

    names = {name for _, _, name in hits}
    assert "bare signal.SIGKILL" in names
    assert "bare os.killpg" in names
    assert "bare os.getuid / os.geteuid / os.getgid" in names
    assert "bare os.setsid" in names


def test_alias_and_canonical_spellings_flag_identically(linter, tmp_path):
    """The regression: these two counts used to be 5 and 0."""
    aliased = _scan(linter, tmp_path / "a", ALIASED)
    canonical = _scan(linter, tmp_path / "b", CANONICAL)

    assert [(n, name) for n, _, name in aliased] == [(n, name) for n, _, name in canonical]
    assert len(aliased) > 0


def test_reported_line_is_the_real_source_not_the_normalized_text(linter, tmp_path):
    """Normalisation is for matching only; output must show what the file says."""
    hits = _scan(linter, tmp_path, ALIASED)

    lines = [line for _, line, _ in hits]
    assert any("_signal.SIGKILL" in line for line in lines)
    assert not any("signal.SIGKILL" in line and "_signal" not in line for line in lines)


def test_suppression_marker_still_wins_on_an_aliased_line(linter, tmp_path):
    source = (
        "import signal as _signal\n"
        "\n"
        "\n"
        "def reap(pid):\n"
        "    import os\n"
        "    os.kill(pid, _signal.SIGKILL)  # windows-footgun: ok -- POSIX arm\n"
    )

    assert _scan(linter, tmp_path, source) == []


def test_guard_hint_is_recognised_through_the_alias(linter, tmp_path):
    """`getattr(_signal, 'SIGKILL', ...)` is as guarded as the canonical form."""
    source = (
        "import signal as _signal\n"
        "\n"
        "\n"
        "def reap():\n"
        "    return getattr(_signal, 'SIGKILL', _signal.SIGTERM)\n"
    )

    assert _scan(linter, tmp_path, source) == []


def test_full_repo_scan_is_clean():
    """`--all` is the CI contract, and widening the linter's reach must not red it.

    Exercised through the script's own entry point rather than scan_file, because
    `--all` picks its own roots and that selection is part of what CI runs.
    """
    result = subprocess.run(
        [sys.executable, str(LINTER_PATH), "--all"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )

    assert result.returncode == 0, result.stdout + result.stderr

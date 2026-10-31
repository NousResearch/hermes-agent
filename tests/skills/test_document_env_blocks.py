"""The side-environment setup shipped in the document skills: the shell blocks and scripts/side_env.py.

Each block picks ``$PY`` for the helper scripts: Hermes's PM environment when ``hermes`` is on
PATH, otherwise ``scripts/side_env.py``, which builds and publishes a venv under ``$HOME/.cache``
(a Docker sandbox has no Hermes). Blocks run under bash against stub interpreters and the helper
runs with a stub venv builder, so these tests need no network and no real pip.
"""
from pathlib import Path
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import tomllib

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOLS = ("bash", "readlink", "chmod", "cat")
HELPERS = [REPO / skill / "scripts" / "side_env.py" for skill in (
    "skills/productivity/pdf", "skills/productivity/microsoft-office",
    "optional-skills/finance/excel-author", "optional-skills/finance/pptx-author")]
BLOCKS = [  # (document, the side_env.py preset its sandbox branch asks for)
    ("skills/productivity/microsoft-office/SKILL.md", "documents"),
    ("skills/productivity/pdf/SKILL.md", "documents"),
    ("optional-skills/finance/excel-author/SKILL.md", "documents"),
    ("optional-skills/finance/pptx-author/SKILL.md", "documents"),
    ("skills/productivity/pdf/references/ocr-extraction.md", "pdf-pymupdf"),
    ("skills/productivity/pdf/references/ocr-extraction.md", "pdf-marker"),
]

pytestmark = [
    pytest.mark.platforms("posix"),  # the blocks are POSIX shell, the sandbox venv layout is bin/python
    pytest.mark.skipif(not all(map(shutil.which, TOOLS)), reason="a POSIX tool the blocks use is missing"),
]


def _helper():
    """A fresh copy of the helper module, so monkeypatching one test never leaks into another."""
    spec = importlib.util.spec_from_file_location("side_env_under_test", HELPERS[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup_block(document, preset):
    """The setup lines of the fenced block whose sandbox branch asks for ``preset``."""
    for chunk in (REPO / document).read_text(encoding="utf-8").split("```bash\n")[1:]:
        code = textwrap.dedent(chunk.split("```", 1)[0])
        if f"side_env.py {preset})" not in code:
            continue
        lines = code.splitlines()
        start = next(i for i, line in enumerate(lines)
                     if line.startswith(('python -c "import', "if command -v hermes")))
        end = max(i for i, line in enumerate(lines) if line in ("fi", "export PY"))
        return "\n".join(lines[start:end + 1]) + "\n"
    raise AssertionError(f"{document}: no setup block asks side_env.py for {preset}")


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _venv(directory, healthy):
    """A stand-in venv whose interpreter passes (or fails) every import check."""
    (directory / "bin").mkdir(parents=True)
    _script(directory / "bin" / "python", "exit 0\n" if healthy else "exit 1\n")


def _run_block(document, preset, tmp_path, *, hermes=None, cwd=None):
    """Run the block (from the skill's directory unless ``cwd``); ``python`` lacks every library,
    ``python3`` is real. Returns the finished process; the block itself must succeed."""
    system_path = os.pathsep.join(dict.fromkeys(os.path.dirname(shutil.which(tool)) for tool in TOOLS))
    if shutil.which("hermes", path=system_path):
        pytest.skip("a hermes on the system PATH would take the PM branch")
    home, stubs = tmp_path / "home", tmp_path / "stubs"
    home.mkdir(exist_ok=True)
    stubs.mkdir()
    _script(stubs / "python", "exit 1\n")
    (stubs / "python3").symlink_to(sys.executable)
    if hermes is not None:
        _script(stubs / "hermes", f"echo {hermes}\n")
    env = {"HOME": str(home), "PATH": f"{stubs}{os.pathsep}{system_path}", "LC_ALL": "C"}
    result = subprocess.run([shutil.which("bash"), "-c", _setup_block(document, preset) + "export -p\n"],
                            env=env, cwd=cwd or ((REPO / document).parent.parent if "references" in document
                                                 else (REPO / document).parent),
                            capture_output=True, check=False, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("document,preset", BLOCKS)
def test_block_exports_py_for_the_next_terminal_call(tmp_path, document, preset):
    # The terminal tool carries only `export -p` into the next call, so an unexported PY is
    # empty there and every `"$PY" scripts/...` line fails.
    exported = _run_block(document, preset, tmp_path, hermes="/side-env/bin/python").stdout
    assert "PY=" in exported and "/side-env/bin/python" in exported, exported


@pytest.mark.parametrize("document,preset", BLOCKS)
def test_sandbox_branch_takes_the_interpreter_side_env_prints(tmp_path, document, preset):
    name = _helper().PRESETS[preset][0]
    cache = tmp_path / "home" / ".cache"
    _venv(cache / f"{name}.PUBLSH", healthy=True)
    (cache / name).symlink_to(f"{name}.PUBLSH")
    result = _run_block(document, preset, tmp_path)
    assert f'PY="{cache / name / "bin" / "python"}"' in result.stdout, result.stdout
    assert "skill directory" not in result.stderr


@pytest.mark.parametrize("document,preset", BLOCKS)
def test_sandbox_branch_from_the_wrong_directory_says_where_to_run(tmp_path, document, preset):
    # The model is not told the in-sandbox skill path; from elsewhere the helper is missing.
    (tmp_path / "elsewhere").mkdir()
    result = _run_block(document, preset, tmp_path, cwd=tmp_path / "elsewhere")
    assert "run from the skill directory" in result.stderr, result.stderr


def test_every_skill_ships_the_same_helper():
    assert {path.read_bytes() for path in HELPERS} == {HELPERS[0].read_bytes()}


def test_documents_preset_pins_are_the_documents_extras():
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert list(_helper().PRESETS["documents"][1]) == extras["documents"] + extras["documents-pdf"]


@pytest.mark.parametrize("preset", ["pdf-pymupdf", "pdf-marker"])
def test_ocr_presets_match_the_hermes_branch(preset):
    block = _setup_block("skills/productivity/pdf/references/ocr-extraction.md", preset)
    pm_line = next(line for line in block.splitlines() if "hermes pm environment" in line)
    assert tuple(re.findall(r"--requirement '([^']+)'", pm_line)) == _helper().PRESETS[preset][1]


def _fake_builder(side, monkeypatch, healthy=True):
    built = []

    def make_venv(stage, requirements):
        built.append(stage.name)
        (stage / "bin").mkdir()
        _script(stage / "bin" / "python", "exit 0\n" if healthy else "exit 1\n")
    monkeypatch.setattr(side, "_make_venv", make_venv)
    return built


def test_broken_published_env_is_replaced_and_removed(tmp_path, monkeypatch):
    side, cache = _helper(), tmp_path / "cache"  # tmp_path also holds the test HERMES_HOME
    cache.mkdir()
    built = _fake_builder(side, monkeypatch)
    _venv(cache / "hermes-docs-venv.BROKEN", healthy=False)
    (cache / "hermes-docs-venv").symlink_to("hermes-docs-venv.BROKEN")
    assert side.ensure("documents", cache) == cache / "hermes-docs-venv" / "bin" / "python"
    assert os.readlink(cache / "hermes-docs-venv") == built[0]
    assert sorted(p.name for p in cache.iterdir()) == sorted(["hermes-docs-venv", "hermes-docs-venv.lock", *built])


def test_failed_build_publishes_nothing_and_keeps_stdout_empty(tmp_path, monkeypatch, capsys):
    side = _helper()
    _fake_builder(side, monkeypatch, healthy=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert side.main(["documents"]) == 1
    out, err = capsys.readouterr()
    assert out == "" and "do not import" in err
    assert sorted(p.name for p in (tmp_path / ".cache").iterdir()) == ["hermes-docs-venv.lock"]


def test_repairer_that_waited_reuses_the_env_published_meanwhile(tmp_path, monkeypatch):
    # Two repairers of a broken env: the second waits for the first's lock, then must reuse the
    # env the first published instead of building (and swapping in) another one.
    side, cache = _helper(), tmp_path / "cache"  # tmp_path also holds the test HERMES_HOME
    cache.mkdir()
    built = _fake_builder(side, monkeypatch)
    checked, real_check = threading.Event(), side._imports_ok
    monkeypatch.setattr(side, "_imports_ok", lambda *a: (real_check(*a), checked.set())[0])
    result = {}
    with side._lock(cache / "hermes-docs-venv.lock"):  # the first repairer, mid-build
        waiter = threading.Thread(target=lambda: result.update(py=side.ensure("documents", cache)))
        waiter.start()
        assert checked.wait(30)  # the waiter found no usable env and now queues for the lock
        _venv(cache / "hermes-docs-venv.FIRST1", healthy=True)
        (cache / "hermes-docs-venv").symlink_to("hermes-docs-venv.FIRST1")
    waiter.join(30)
    assert not waiter.is_alive()
    assert result == {"py": cache / "hermes-docs-venv" / "bin" / "python"} and built == []
    assert os.readlink(cache / "hermes-docs-venv") == "hermes-docs-venv.FIRST1"


def test_stale_stages_and_old_generations_are_swept(tmp_path, monkeypatch):
    side, cache = _helper(), tmp_path / "cache"  # tmp_path also holds the test HERMES_HOME
    cache.mkdir()
    built = _fake_builder(side, monkeypatch)
    hour_ago = time.time() - side.STALE_AFTER_S - 60
    _venv(cache / "hermes-docs-venv.CURRNT", healthy=True)
    (cache / "hermes-docs-venv").symlink_to("hermes-docs-venv.CURRNT")
    for stale in ("hermes-docs-venv.OLDGEN", "hermes-docs-venv.ABC123"):  # a replaced generation, a crashed stage
        _venv(cache / stale, healthy=True)
    (cache / "hermes-docs-venv.ABC123.lnk").symlink_to("hermes-docs-venv.ABC123")
    _venv(cache / "hermes-docs-venv.YOUNG1", healthy=False)  # may still be building
    _venv(cache / "hermes-pdf-marker-venv.OTHER1", healthy=True)  # another env's generation
    for entry in cache.iterdir():
        if entry.name != "hermes-docs-venv.YOUNG1":
            os.utime(entry, (hour_ago, hour_ago), follow_symlinks=False)
    side.ensure("documents", cache)
    assert built == []
    assert sorted(p.name for p in cache.iterdir()) == [
        "hermes-docs-venv", "hermes-docs-venv.CURRNT", "hermes-docs-venv.YOUNG1", "hermes-docs-venv.lock",
        "hermes-pdf-marker-venv.OTHER1"]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores the 0555 mode")
def test_healthy_env_in_a_read_only_cache_is_still_used(tmp_path, monkeypatch):
    # No .lock can be created in the cache: the best-effort sweep is skipped, the env still works.
    side, cache = _helper(), tmp_path / "cache"
    cache.mkdir()
    built = _fake_builder(side, monkeypatch)
    _venv(cache / "hermes-docs-venv.CURRNT", healthy=True)
    (cache / "hermes-docs-venv").symlink_to("hermes-docs-venv.CURRNT")
    cache.chmod(0o555)
    try:
        assert side.ensure("documents", cache) == cache / "hermes-docs-venv" / "bin" / "python"
    finally:
        cache.chmod(0o755)
    assert built == [] and not (cache / "hermes-docs-venv.lock").exists()


def test_old_stage_survives_while_another_builder_holds_the_lock(tmp_path, monkeypatch):
    # The lock is the only guard: a stage's mtime freezes during pip install, so age alone
    # cannot tell a live build from an abandoned one.
    side, cache = _helper(), tmp_path / "cache"
    cache.mkdir()
    built = _fake_builder(side, monkeypatch)
    _venv(cache / "hermes-docs-venv.CURRNT", healthy=True)
    (cache / "hermes-docs-venv").symlink_to("hermes-docs-venv.CURRNT")
    _venv(cache / "hermes-docs-venv.BUILDG", healthy=False)
    two_hours_ago = time.time() - 2 * 3600
    os.utime(cache / "hermes-docs-venv.BUILDG", (two_hours_ago, two_hours_ago))
    with side._lock(cache / "hermes-docs-venv.lock"):  # another builder, mid pip install
        assert side.ensure("documents", cache) == cache / "hermes-docs-venv" / "bin" / "python"
    assert built == [] and (cache / "hermes-docs-venv.BUILDG").is_dir()


def test_sweep_skips_an_entry_removed_since_the_glob(tmp_path, monkeypatch):
    side, cache = _helper(), tmp_path / "cache"
    cache.mkdir()
    stale = cache / "hermes-docs-venv.OLDGEN"
    _venv(stale, healthy=True)
    hour_ago = time.time() - side.STALE_AFTER_S - 60
    os.utime(stale, (hour_ago, hour_ago))
    real_glob = type(cache).glob
    monkeypatch.setattr(type(cache), "glob",
                        lambda self, pattern, **kw: [cache / "hermes-docs-venv.GONE00", *real_glob(self, pattern, **kw)])
    side._sweep(cache, "hermes-docs-venv")
    assert not stale.exists()

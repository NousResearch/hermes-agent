"""The side-environment setup blocks shipped in the document skills, run as written.

Each block picks ``$PY`` for the helper scripts: Hermes's PM environment when ``hermes`` is on
PATH, otherwise a venv it builds and publishes under ``$HOME/.cache`` (a Docker sandbox has no
Hermes). Every block is executed by bash against stub interpreters, so these tests need no
network and no real pip.
"""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = ("bash", "mkdir", "mktemp", "readlink", "ln", "rm", "chmod", "cat")
BLOCKS = [  # (document, the venv name its sandbox branch publishes under $HOME/.cache)
    ("skills/productivity/microsoft-office/SKILL.md", "hermes-docs-venv"),
    ("skills/productivity/pdf/SKILL.md", "hermes-docs-venv"),
    ("optional-skills/finance/excel-author/SKILL.md", "hermes-docs-venv"),
    ("optional-skills/finance/pptx-author/SKILL.md", "hermes-docs-venv"),
    ("skills/productivity/pdf/references/ocr-extraction.md", "hermes-pdf-pymupdf-venv"),
    ("skills/productivity/pdf/references/ocr-extraction.md", "hermes-pdf-marker-venv"),
]

pytestmark = [
    pytest.mark.platforms("posix"),  # the blocks are POSIX shell
    pytest.mark.skipif(not all(map(shutil.which, TOOLS)), reason="a POSIX tool the blocks use is missing"),
]


def _system_path():
    """Only the directories of the POSIX tools the blocks use, so no real hermes/python leaks in."""
    return os.pathsep.join(dict.fromkeys(os.path.dirname(shutil.which(tool)) for tool in TOOLS))


def _setup_block(document, venv):
    """The setup lines of the fenced block that publishes ``venv``, from its first line to ``fi``/``export PY``."""
    for chunk in (REPO / document).read_text(encoding="utf-8").split("```bash\n")[1:]:
        code = textwrap.dedent(chunk.split("```", 1)[0])
        if f"readlink {venv})" not in code:
            continue
        lines = code.splitlines()
        start = next(i for i, line in enumerate(lines)
                     if line.startswith(('python -c "import', "if command -v hermes")))
        end = max(i for i, line in enumerate(lines) if line in ("fi", "export PY"))
        return "\n".join(lines[start:end + 1]) + "\n"
    raise AssertionError(f"{document}: no setup block publishes {venv}")


def _script(path, body):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _venv(directory, healthy):
    """A stand-in venv whose interpreter passes (or fails) every import check and pip install."""
    (directory / "bin").mkdir(parents=True)
    _script(directory / "bin" / "python", "exit 0\n" if healthy else "exit 1\n")


def _sandbox(tmp_path, *, hermes=None, before_stage=""):
    """A HOME and a PATH for one block: ``python`` lacks every library and ``python3 -m venv``
    makes a stub venv. ``before_stage`` runs inside the block's ``$HOME/.cache`` right before it
    creates its private stage, i.e. after its own import check failed."""
    system_path = _system_path()
    if shutil.which("hermes", path=system_path):
        pytest.skip("a hermes on the system PATH would take the PM branch")
    home, stubs = tmp_path / "home", tmp_path / "stubs"
    home.mkdir()
    stubs.mkdir()
    _script(stubs / "python", "exit 1\n")
    _script(stubs / "python3", textwrap.dedent(f"""\
        if [ "$1 $2" = "-m venv" ]; then mkdir -p "$3/bin" && printf '#!/bin/sh\\nexit 0\\n' > "$3/bin/python" && chmod +x "$3/bin/python"; exit; fi
        exec {shlex.quote(sys.executable)} "$@"
        """))
    _script(stubs / "mktemp", f"{before_stage}\nexec {shlex.quote(shutil.which('mktemp'))} \"$@\"\n")
    if hermes is not None:
        _script(stubs / "hermes", f"echo {hermes}\n")
    return home, {"HOME": str(home), "PATH": f"{stubs}{os.pathsep}{system_path}", "LC_ALL": "C"}


def _run(script, env, cwd):
    result = subprocess.run([shutil.which("bash"), "-c", script], env=env, cwd=cwd, capture_output=True, check=False,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("document,venv", BLOCKS)
def test_block_exports_py_for_the_next_terminal_call(tmp_path, document, venv):
    # The terminal tool carries only `export -p` into the next call, so an unexported PY is
    # empty there and every `"$PY" scripts/...` line fails.
    home, env = _sandbox(tmp_path, hermes="/side-env/bin/python")
    exported = _run(_setup_block(document, venv) + "export -p\n", env, home)
    assert "PY=" in exported and "/side-env/bin/python" in exported, exported


@pytest.mark.parametrize("document,venv", BLOCKS)
def test_late_builder_keeps_the_env_another_caller_just_published(tmp_path, document, venv):
    # Another caller publishes a working build between this caller's failed import check and
    # its stage: this caller must not replace it and delete it under that caller.
    published = f"{venv}.PUBLSH"
    publish = (f'[ -e {published} ] || {{ mkdir -p {published}/bin && printf "#!/bin/sh\\nexit 0\\n" '
               f'> {published}/bin/python && chmod +x {published}/bin/python && ln -s {published} {venv}; }}')
    home, env = _sandbox(tmp_path, before_stage=publish)
    _run(_setup_block(document, venv), env, home)
    cache = home / ".cache"
    assert (cache / published / "bin" / "python").is_file()
    assert os.readlink(cache / venv) == published
    assert sorted(p.name for p in cache.iterdir()) == [venv, published]


@pytest.mark.parametrize("document,venv", BLOCKS)
def test_broken_published_env_is_replaced_and_removed(tmp_path, document, venv):
    home, env = _sandbox(tmp_path)
    cache = home / ".cache"
    _venv(cache / f"{venv}.BROKEN", healthy=False)
    (cache / venv).symlink_to(f"{venv}.BROKEN")
    _run(_setup_block(document, venv), env, home)
    fresh = os.readlink(cache / venv)
    assert fresh.startswith(f"{venv}.") and fresh != f"{venv}.BROKEN"
    assert sorted(p.name for p in cache.iterdir()) == sorted([venv, fresh])


def test_repair_still_works_when_the_test_tools_live_under_a_spaced_path(tmp_path, monkeypatch):
    # The stubs exec the real interpreter and mktemp; a checkout or venv under a spaced path
    # must not split those commands into words (the blocks themselves are not involved).
    spaced = tmp_path / "test tools"
    spaced.mkdir()
    (spaced / "python").symlink_to(sys.executable)
    (spaced / "mktemp").symlink_to(shutil.which("mktemp"))
    real_which = shutil.which
    monkeypatch.setattr(sys, "executable", str(spaced / "python"))
    monkeypatch.setattr(shutil, "which",
                        lambda cmd, *a, **k: str(spaced / "mktemp") if cmd == "mktemp" else real_which(cmd, *a, **k))
    test_broken_published_env_is_replaced_and_removed(tmp_path, *BLOCKS[1])

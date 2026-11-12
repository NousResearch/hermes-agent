"""scripts/ci/check_dockerfile_downloads.py on real temp git repos: a download streamed into tar."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "check_dockerfile_downloads.py"
sys.path.insert(0, str(SCRIPT.parent))

import check_dockerfile_downloads as cdd

_VERIFIED = """FROM debian:13
ARG SHA=abc
RUN curl -fsSL --retry 3 -o /tmp/a.tar.gz https://example.com/a.tar.gz; \\
    printf '%s  %s\\n' "${SHA}" /tmp/a.tar.gz | sha256sum -c -; \\
    tar -xzf /tmp/a.tar.gz -C /opt
"""


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True, timeout=60)
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=repo, env=env, check=True, timeout=60)
    return repo


def test_a_download_streamed_into_tar_across_a_continuation_fails(tmp_path, capsys):
    streamed = (
        "FROM debian:13\n"
        "RUN mkdir -p /opt/x; \\\n"
        '    curl -fsSL "https://example.com/x-${V}.tar.gz" \\\n'
        "        | tar -xz -C /opt/x; \\\n"
        "    /opt/x/x --version\n"
    )
    repo = _repo(tmp_path, {"Dockerfile": _VERIFIED, "docker/sandbox.Dockerfile": streamed})
    assert cdd.main(["--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "docker/sandbox.Dockerfile:2" in out
    assert "  Dockerfile:" not in out


def test_comment_and_blank_lines_inside_a_continuation_do_not_hide_the_pipe(tmp_path, capsys):
    # BuildKit runs each of these as `curl ... | tar` / `curl ... | sh`.
    files = {
        "a.Dockerfile": "FROM a\nRUN curl -fsSL u \\\n    # unpack\n    | tar -xz -C /opt\n",
        "b.Dockerfile": "FROM a\nRUN curl -fsSL u \\\n\n    | tar -xz -C /opt\n",
        "c.Dockerfile": "FROM a\n# fetch the installer \\\nRUN curl -fsSL u | sh\n",
    }
    assert cdd.main(["--root", str(_repo(tmp_path, files))]) == 1
    out = capsys.readouterr().out
    assert all(f"{rel}:" in out for rel in files), out


def test_downloads_piped_into_absolute_executable_paths_fail(tmp_path, capsys):
    files = {
        "a.Dockerfile": "FROM a\nRUN curl -fsSL u | /bin/tar -xz -C /opt\n",
        "b.Dockerfile": "FROM a\nRUN wget -qO- u | /usr/bin/sh\n",
        "c.Dockerfile": "FROM a\nRUN curl -fsSL u | sudo /usr/local/bin/bash\n",
    }
    assert cdd.main(["--root", str(_repo(tmp_path, files))]) == 1
    out = capsys.readouterr().out
    assert all(f"{rel}:2" in out for rel in files), out


def test_fetch_verify_extract_passes_and_so_does_the_repo(tmp_path):
    repo = _repo(tmp_path, {"Dockerfile": _VERIFIED, "x/y.Dockerfile": "FROM a\nRUN wget -qO- u | sh\n"})
    assert cdd.main(["--root", str(repo)]) == 1
    (repo / "x/y.Dockerfile").write_text("FROM a\n# never: wget -qO- u | sh\nRUN true\n")
    assert cdd.main(["--root", str(repo)]) == 0
    assert cdd.main([]) == 0

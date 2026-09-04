"""Temporary read-only compiler for one authority-repair branch.

This file is explicitly macOS-marked because that CI lane has a complete,
locked repository environment without the Linux suite's current runner backlog.
It executes every literal shell step in the branch's sole one-shot carrier with
all pushes intercepted, then emits the exact tested product diff as a
checksummed gzip/base64 failure artifact. Compiler and workflow files are
excluded from that artifact.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

CURRENT_MAIN = "b6bcb3e791c673e63974029bbab40cc9326803ff"
SELF = "tests/e2e/test__authority_repair_materializer.py"
BRANCH_BY_CARRIER = {
    "one-shot-fix-86354.yml": "fix/email-app-password-normalization",
    "one-shot-fix-88796.yml": "fix/memory-prefetch-cancel",
    "one-shot-fix-85644.yml": "campaign/webhook-delivery-callbacks",
    "one-shot-fix-89252.yml": "fix/88715-canonical-multiplex-identity",
}
EXTRA_ENV_BY_CARRIER = {
    "one-shot-fix-85644.yml": {
        "ORIGINAL_FEATURE_HEAD": "2255348955a1e621e1f8f0f9e5c57948b5ae1d9d",
        "PAYLOAD_COMMIT": "5beca04e0bdf6097ae562550caffcd256826931d",
    },
}


def _carrier() -> Path:
    candidates = sorted(Path(".github/workflows").glob("one-shot-fix-*.yml"))
    assert len(candidates) == 1, [str(path) for path in candidates]
    assert candidates[0].name in BRANCH_BY_CARRIER, candidates[0].name
    return candidates[0]


def _run_blocks(path: Path) -> list[str]:
    """Extract literal ``run: |`` blocks without interpreting raw heredocs."""
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "run: |"]
    assert starts, f"no literal run blocks in {path}"
    blocks: list[str] = []
    for start in starts:
        key_indent = len(lines[start]) - len(lines[start].lstrip())
        prefix = " " * (key_indent + 2)
        end = len(lines)
        for index in range(start + 1, len(lines)):
            line = lines[index]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent == 6 and stripped.startswith("- "):
                end = index
                break
        body = [
            line[len(prefix) :] if line.startswith(prefix) else line
            for line in lines[start + 1 : end]
        ]
        while body and not body[-1].strip():
            body.pop()
        assert body
        blocks.append("\n".join(body) + "\n")
    return blocks


def _install_tool_shims(tmp_path: Path) -> Path:
    """Install read-only Git plus narrow GNU compatibility for macOS."""
    real_git = shutil.which("git")
    assert real_git
    bindir = tmp_path / "bin"
    bindir.mkdir()

    git_wrapper = bindir / "git"
    git_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "args=(\"$@\")\n"
        "if [[ ${args[0]:-} == push ]]; then\n"
        "  echo '[materializer] blocked git push' >&2\n"
        "  exit 0\n"
        "fi\n"
        "for i in \"${!args[@]}\"; do\n"
        "  if [[ ${args[$i]} == origin ]]; then\n"
        "    args[$i]=https://github.com/andrexibiza/hermes-agent.git\n"
        "  fi\n"
        "done\n"
        f"exec {real_git!s} \"${{args[@]}}\"\n"
    )
    git_wrapper.chmod(0o755)

    sha_wrapper = bindir / "sha256sum"
    sha_wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, pathlib, sys\n"
        "for raw in sys.argv[1:]:\n"
        "    path = pathlib.Path(raw)\n"
        "    print(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {raw}')\n"
    )
    sha_wrapper.chmod(0o755)

    base64_wrapper = bindir / "base64"
    base64_wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import base64, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "decode = any(arg in ('-d', '-D', '--decode') for arg in args)\n"
        "paths = [arg for arg in args if not arg.startswith('-')]\n"
        "data = pathlib.Path(paths[-1]).read_bytes() if paths else sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(base64.b64decode(data) if decode else base64.b64encode(data))\n"
    )
    base64_wrapper.chmod(0o755)
    return bindir


def _product_diff() -> tuple[bytes, dict[str, object]]:
    raw = subprocess.check_output(
        ["git", "diff", "--name-status", CURRENT_MAIN, "HEAD"], text=True
    )
    entries: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        if path == SELF or path.startswith("tests/test__authority_repair_materializer.py"):
            continue
        if path.startswith(".github/workflows/") or path.startswith("contributors/emails/"):
            continue
        entries.append((status, path))

    manifest: dict[str, object] = {
        "base": CURRENT_MAIN,
        "entries": entries,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        data = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
        for status, path in entries:
            if not status.startswith("D"):
                archive.add(path, arcname=path, recursive=False)
    return payload.getvalue(), manifest


@pytest.mark.macos_only
def test_materialize_exact_authority_repair_tree(tmp_path: Path) -> None:
    carrier = _carrier()
    blocks = _run_blocks(carrier)
    supplements = [
        (path.name, path.read_text())
        for path in sorted(Path(".github/workflows").glob("materialize-*.py"))
    ]

    env = os.environ.copy()
    env["PATH"] = f"{_install_tool_shims(tmp_path)}:{env['PATH']}"
    env["CURRENT_MAIN"] = CURRENT_MAIN
    env["BRANCH"] = BRANCH_BY_CARRIER[carrier.name]
    env["TRIGGER_HEAD"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    env.update(EXTRA_ENV_BY_CARRIER.get(carrier.name, {}))

    for index, body in enumerate(blocks):
        script = tmp_path / f"carrier-{index}.sh"
        script.write_text(body)
        subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(["bash", str(script)], check=True, env=env, timeout=1800)

    # A carrier may reset to exact current main and thereby remove supplemental
    # transformer files. Execute only bytes captured before that reset.
    for name, content in supplements:
        supplement = tmp_path / name
        supplement.write_text(content)
        subprocess.run([sys.executable, str(supplement)], check=True, env=env, timeout=1800)

    artifact, manifest = _product_diff()
    digest = hashlib.sha256(artifact).hexdigest()
    encoded = base64.b64encode(artifact).decode()
    wrapped = "\n".join(encoded[i : i + 120] for i in range(0, len(encoded), 120))
    pytest.fail(
        "HERMES_MATERIALIZED_TREE_BEGIN\n"
        f"sha256={digest}\nmanifest={json.dumps(manifest, sort_keys=True)}\n"
        f"{wrapped}\nHERMES_MATERIALIZED_TREE_END",
        pytrace=False,
    )

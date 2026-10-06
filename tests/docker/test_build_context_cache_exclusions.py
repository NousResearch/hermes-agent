"""Docker build contexts exclude Python caches and retain build inputs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_build_context_excludes_python_caches_and_preserves_build_inputs(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    policy = (source_root / ".dockerignore").read_bytes()
    required_inputs = {
        "ordinary/nested/module.py": b"pass\n",
        "scripts/build/icon_environment.py": (
            source_root / "scripts/build/icon_environment.py"
        ).read_bytes(),
        "scripts/build/build_frontend.py": b"pass\n",
        "scripts/build/product.so": b"build product\n",
        "apps/shared/index.js": b"export {};\n",
        "apps/shared/native.node": b"native product\n",
        "assets/nous-girl-black.svg": b"<svg/>\n",
        "assets/nous-girl-white.svg": b"<svg/>\n",
        "assets/dmg-volume.png": b"source artwork\n",
        "assets/backgrounds/test.png": b"background artwork\n",
        ".env.example": b"EXAMPLE=value\n",
    }
    cache_paths = (
        "module.pyc",
        "__pycache__/module.pyc",
        "ordinary/nested/__pycache__/module.cpython-314.pyc",
        "ordinary/nested/module.pyc",
        "ordinary/nested/module.pyo",
        "scripts/build/__pycache__/icon_environment.cpython-314.pyc",
        "scripts/build/nested/cache.pyc",
        "scripts/build/nested/cache.pyo",
        "apps/shared/__pycache__/module.cpython-314.pyc",
        "apps/shared/nested/cache.pyc",
        "apps/shared/nested/cache.pyo",
    )
    excluded_inputs = {
        ".env": b"SECRET=sentinel\n",
        ".venv/placeholder": b"virtual environment\n",
        "node_modules/placeholder": b"dependency\n",
        "tests/placeholder.py": b"pass\n",
        "runtime/placeholder": b"runtime state\n",
        "assets/unused.svg": b"<svg/>\n",
        **dict.fromkeys(cache_paths, b"cached bytecode\n"),
    }
    context = tmp_path / "context"
    context.mkdir()
    (context / ".dockerignore").write_bytes(policy)
    for relative_path, contents in {**required_inputs, **excluded_inputs}.items():
        destination = context / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)

    recipe = tmp_path / "filter.Dockerfile"
    recipe.write_text("FROM scratch\nCOPY . /context/\n", encoding="utf-8")
    export = tmp_path / "export"
    result = subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            "--network=none",
            "--provenance=false",
            "--sbom=false",
            "--output",
            f"type=local,dest={export},platform-split=false",
            "-f",
            str(recipe),
            str(context),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    actual = {
        path.relative_to(export / "context").as_posix(): path.read_bytes()
        for path in (export / "context").rglob("*")
        if path.is_file()
    }
    assert actual == {".dockerignore": policy, **required_inputs}

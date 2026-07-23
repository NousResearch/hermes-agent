from __future__ import annotations

from pathlib import Path

from scripts.termux_requirements import expand_termux_requirements, lock_constraints


def test_expands_profile_and_removes_native_optional_extras(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "hermes-agent"
dependencies = [
  "uvicorn[standard]>=0.24,<1",
  "basepkg==1",
  "windows-only==1; sys_platform == 'win32'",
]
[project.optional-dependencies]
termux-all = ["hermes-agent[termux]", "hermes-agent[web]"]
termux = ["python-telegram-bot[webhooks]==22.6", "hermes-agent[cli]"]
cli = ["simple-term-menu==1.6.6"]
web = ["uvicorn[standard]==0.41.0", "fastapi==1.0"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    requirements = expand_termux_requirements(pyproject, python_version="3.11.15")
    assert "uvicorn<1,>=0.24" in requirements
    assert "uvicorn==0.41.0" in requirements
    assert "python-telegram-bot==22.6" in requirements
    assert "simple-term-menu==1.6.6" in requirements
    assert all("standard" not in value for value in requirements)
    assert all("webhooks" not in value for value in requirements)
    assert all("windows-only" not in value for value in requirements)


def test_lock_constraints_skip_platform_forks(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """
version = 1
[[package]]
name = "single"
version = "1.2.3"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "forked"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "forked"
version = "2.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "local"
version = "0.1"
source = { editable = "." }
""".strip()
        + "\n",
        encoding="utf-8",
    )
    constraints = lock_constraints(lock)
    assert constraints == ["single==1.2.3"]

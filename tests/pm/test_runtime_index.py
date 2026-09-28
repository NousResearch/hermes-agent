"""The PM runtime preserves lock hashes while installing through a custom index."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from pm.package import InstallError
from pm.runtime_stage import stage_runtime
from tests.pm._range_server import RangeHandler


def _wheel(name: str, version: str, modules: dict[str, bytes]) -> tuple[str, bytes]:
    normalized = re.sub(r"[-_.]+", "_", name).lower()
    dist_info = f"{normalized}-{version}.dist-info"
    files = {
        **modules,
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: hermes-agent-tests\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ).encode(),
    }
    rows = []
    for path, body in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        rows.append((path, f"sha256={digest}", str(len(body))))
    record = f"{dist_info}/RECORD"
    rows.append((record, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    files[record] = output.getvalue().encode()
    filename = f"{normalized}-{version}-py3-none-any.whl"
    stream = io.BytesIO()
    with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
        for path, body in files.items():
            archive.writestr(path, body)
    return filename, stream.getvalue()


@pytest.fixture
def package_indexes():
    class Handler(RangeHandler):
        pages: dict[str, bytes] = {}

        def _respond(self, body: bytes, *, include_body: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=UTF-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            page = self.pages.get(self.path.split("?", 1)[0])
            if page is None:
                return super().do_GET()
            self._respond(page, include_body=True)

        def do_HEAD(self):  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            body = self.pages.get(path) or self.payloads.get(path)
            if body is None:
                return self.send_error(404)
            self._respond(body, include_body=False)

    Handler.payloads = {}
    Handler.pages = {}
    Handler.ranges_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _index(server: ThreadingHTTPServer, handler: type[RangeHandler], prefix: str,
           packages: dict[str, tuple[str, bytes]]) -> str:
    base = f"http://127.0.0.1:{server.server_port}"
    for name, (filename, wheel) in packages.items():
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        wheel_path = f"/{prefix}/files/{filename}"
        handler.payloads[wheel_path] = wheel
        digest = hashlib.sha256(wheel).hexdigest()
        page = f'<a href="{base}{wheel_path}#sha256={digest}">{filename}</a>'.encode()
        for suffix in ("", "/"):
            handler.pages[f"/{prefix}/simple/{canonical}{suffix}"] = page
    return f"{base}/{prefix}/simple"


def _lock_project(uv: str, project: Path, lock_index: str) -> tuple[Path, bytes]:
    clean_env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("PIP_", "UV_"))
    }
    clean_env["PIP_CONFIG_FILE"] = os.devnull
    clean_env["NO_PROXY"] = "127.0.0.1,localhost"
    clean_env["UV_INDEX_URL"] = lock_index
    locked = subprocess.run(
        [uv, "lock", "--python", sys.executable, "--no-config"],
        cwd=project, env=clean_env, capture_output=True, text=True, timeout=120,
    )
    assert locked.returncode == 0, locked.stdout + locked.stderr
    lock_path = project / "uv.lock"
    lock_text = lock_path.read_text(encoding="utf-8")
    assert lock_index in lock_text
    lock_path.write_text(
        lock_text.replace(lock_index, "https://pypi.org/simple"),
        encoding="utf-8",
    )
    return lock_path, hashlib.sha256(lock_path.read_bytes()).digest()


def _write_buildable_project(
    directory: Path,
    *,
    project_name: str,
    module_name: str,
    value: str,
    dependencies: tuple[str, ...] = (),
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    module = directory / module_name
    module.mkdir()
    (module / "__init__.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")
    (directory / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\nbuild-backend = \"backend\"\nbackend-path = [\".\"]\n\n"
        "[project]\n"
        f"name = {json.dumps(project_name)}\n"
        'version = "1.0.0"\n'
        'requires-python = ">=3.11,<3.15"\n'
        f"dependencies = {json.dumps(dependencies)}\n",
        encoding="utf-8",
    )
    dist_name = project_name.replace("-", "_")
    backend = f'''from base64 import urlsafe_b64encode
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

NAME = {project_name!r}
DIST = {dist_name!r}
MODULE = {module_name!r}
VALUE = {value!r}
VERSION = "1.0.0"
DEPENDENCIES = {dependencies!r}

def _build(wheel_directory, *, editable=False):
    filename = f"{{DIST}}-{{VERSION}}-py3-none-any.whl"
    dist_info = f"{{DIST}}-{{VERSION}}.dist-info"
    requires = "".join(f"Requires-Dist: {{dependency}}\\n" for dependency in DEPENDENCIES)
    files = {{
        f"{{dist_info}}/METADATA": (
            f"Metadata-Version: 2.1\\nName: {{NAME}}\\nVersion: {{VERSION}}\\n"
            f"Requires-Python: >=3.11,<3.15\\n{{requires}}\\n"
        ).encode(),
        f"{{dist_info}}/WHEEL": (
            "Wheel-Version: 1.0\\nGenerator: hermes-agent-tests\\n"
            "Root-Is-Purelib: true\\nTag: py3-none-any\\n"
        ).encode(),
    }}
    if editable:
        files[f"{{MODULE}}.pth"] = (str(Path(__file__).parent) + "\\n").encode()
    else:
        files[f"{{MODULE}}/__init__.py"] = f"VALUE = {{VALUE!r}}\\n".encode()
    rows = []
    for path, body in files.items():
        digest = urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        rows.append((path, f"sha256={{digest}}", str(len(body))))
    record = f"{{dist_info}}/RECORD"
    rows.append((record, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\\n").writerows(rows)
    files[record] = output.getvalue().encode()
    with ZipFile(Path(wheel_directory) / filename, "w", ZIP_DEFLATED) as archive:
        for path, body in files.items():
            archive.writestr(path, body)
    return filename

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(wheel_directory)

def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(wheel_directory, editable=True)
'''
    (directory / "backend.py").write_text(backend, encoding="utf-8")


def test_runtime_installs_mirror_artifacts_from_locked_hashes(tmp_path, monkeypatch, package_indexes):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the real dependency-runtime test")

    server, handler = package_indexes
    packages = {
        "packaging": ("26.0", {"packaging/__init__.py": b"__version__ = '26.0'\n"}),
        "tomli-w": ("1.2.0", {"tomli_w/__init__.py": b""}),
        "truststore": ("0.10.4", {"truststore/__init__.py": b""}),
        "ruamel.yaml": (
            "0.18.16",
            {
                "ruamel/__init__.py": b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n",
                "ruamel/yaml/__init__.py": b"class YAML: pass\n",
            },
        ),
    }
    wheels = {name: _wheel(name, version, modules) for name, (version, modules) in packages.items()}
    lock_index = _index(server, handler, "locked", wheels)
    mirror_index = _index(server, handler, "mirror", wheels)
    project = tmp_path / "project"
    project.mkdir()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dependencies = ",\n".join(f'"{name}=={version}"' for name, (version, _) in packages.items())
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "pm-runtime-index-fixture"\nversion = "0.0.0"\n'
        f'requires-python = ">={python_version},<{sys.version_info.major}.{sys.version_info.minor + 1}"\n'
        f"dependencies = [{dependencies}]\n"
        "[tool.uv]\npackage = false\n"
        "[tool.uv.workspace]\nmembers = []\n",
        encoding="utf-8",
    )

    lock_path, lock_digest = _lock_project(uv, project, lock_index)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    for key in list(os.environ):
        if key.startswith(("PIP_", "UV_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PIP_INDEX_URL", mirror_index)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    python = stage_runtime(
        Path(uv), Path(sys.executable), tmp_path / "runtime",
        project=project, cache=tmp_path / "uv-cache",
    )
    checked = subprocess.run(
        [str(python), "-I", "-B", "-c",
         "import packaging, tomli_w, truststore; from ruamel.yaml import YAML"],
        capture_output=True, text=True, timeout=30,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert hashlib.sha256(lock_path.read_bytes()).digest() == lock_digest

    from pm.environment import PythonEnvironment
    from pm.runtime import runtime_environment

    environment = PythonEnvironment(
        uv=Path(uv),
        python=Path(sys.executable),
        destination=tmp_path / "synced-runtime",
        cache=tmp_path / "uv-cache",
        env=runtime_environment(),
        no_config=True,
    )
    environment.create()
    environment.sync(project, no_default_groups=True, no_install_project=True)
    checked_sync = subprocess.run(
        [str(environment.executable), "-I", "-B", "-c",
         "import packaging, tomli_w, truststore; from ruamel.yaml import YAML"],
        capture_output=True, text=True, timeout=30,
    )
    assert checked_sync.returncode == 0, checked_sync.stdout + checked_sync.stderr

    bad_packages = dict(wheels)
    bad_packages["packaging"] = _wheel(
        "packaging", "26.0", {"packaging/__init__.py": b"__version__ = 'modified'\n"},
    )
    bad_index = _index(server, handler, "bad-mirror", bad_packages)
    monkeypatch.setenv("PIP_INDEX_URL", bad_index)
    with pytest.raises(InstallError, match="(?i)hash"):
        stage_runtime(
            Path(uv), Path(sys.executable), tmp_path / "tampered-runtime",
            project=project, cache=tmp_path / "uv-cache",
        )


def test_mirror_sync_defers_workspace_and_pinned_git_sources(tmp_path, monkeypatch, package_indexes):
    uv = shutil.which("uv")
    git = shutil.which("git")
    if uv is None or git is None:
        pytest.skip("uv and git are required for the real dependency-sync test")

    server, handler = package_indexes
    workspace_wheel = _wheel(
        "workspace-dep", "1.0.0", {"workspace_dep/__init__.py": b"VALUE = 'workspace dep'\n"},
    )
    git_wheel = _wheel(
        "git-dep", "1.0.0", {"git_dep/__init__.py": b"VALUE = 'git dep'\n"},
    )
    packages = {"workspace-dep": workspace_wheel, "git-dep": git_wheel}
    lock_index = _index(server, handler, "locked", packages)
    mirror_index = _index(server, handler, "mirror", packages)

    project = tmp_path / "project"
    project.mkdir()
    _write_buildable_project(
        project / "plugin",
        project_name="workspace-plugin",
        module_name="workspace_plugin",
        value="workspace",
        dependencies=("workspace-dep==1.0.0",),
    )
    git_project = tmp_path / "git-fixture"
    _write_buildable_project(
        git_project,
        project_name="git-fixture",
        module_name="git_fixture",
        value="git",
        dependencies=("git-dep==1.0.0",),
    )
    for command in (
        [git, "init", "--quiet"],
        [git, "add", "--all"],
        [git, "-c", "user.name=Hermes Tests", "-c", "user.email=tests@example.com",
         "commit", "--quiet", "-m", "fixture"],
    ):
        completed = subprocess.run(
            command, cwd=git_project, capture_output=True, text=True, timeout=30,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    commit = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=git_project,
        capture_output=True, text=True, timeout=30,
    )
    assert commit.returncode == 0, commit.stdout + commit.stderr
    git_requirement = f"git-fixture @ git+{git_project.as_uri()}@{commit.stdout.strip()}"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    (project / "pyproject.toml").write_text(
        '[project]\nname = "source-graph-fixture"\nversion = "0.0.0"\n'
        f'requires-python = ">={python_version},<{sys.version_info.major}.{sys.version_info.minor + 1}"\n'
        f"dependencies = [\"workspace-plugin\", {json.dumps(git_requirement)}]\n"
        "[tool.uv]\npackage = false\n"
        "[tool.uv.sources]\nworkspace-plugin = { workspace = true }\n"
        "[tool.uv.workspace]\nmembers = [\"plugin\"]\n",
        encoding="utf-8",
    )
    lock_path, lock_digest = _lock_project(uv, project, lock_index)

    for key in list(os.environ):
        if key.startswith(("PIP_", "UV_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PIP_INDEX_URL", mirror_index)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    from pm.environment import PythonEnvironment
    from pm.runtime import runtime_environment

    hashed_requirements: list[str] = []
    install_requirements = PythonEnvironment._install_requirements_file

    def capture_hashed_requirements(self, requirements, **kwargs):
        if kwargs.get("require_hashes"):
            hashed_requirements.append(requirements.read_text(encoding="utf-8"))
        return install_requirements(self, requirements, **kwargs)

    monkeypatch.setattr(
        PythonEnvironment, "_install_requirements_file", capture_hashed_requirements,
    )
    environment = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "venv",
        cache=tmp_path / "uv-cache", env=runtime_environment(), no_config=True,
    )
    environment.create()
    environment.sync(project, no_default_groups=True)

    assert len(hashed_requirements) == 1
    requirements = hashed_requirements[0].lower()
    assert "workspace-dep==1.0.0" in requirements
    assert "git-dep==1.0.0" in requirements
    assert "--hash=sha256:" in requirements
    assert "workspace-plugin" not in requirements
    assert "git+" not in requirements
    assert hashlib.sha256(lock_path.read_bytes()).digest() == lock_digest
    imported = subprocess.run(
        [str(environment.executable), "-I", "-B", "-c",
         "import git_dep, git_fixture, workspace_dep, workspace_plugin; "
         "assert git_fixture.VALUE == 'git' and workspace_plugin.VALUE == 'workspace'"],
        capture_output=True, text=True, timeout=30,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr


def test_hash_only_install_rejects_requirements_without_a_hash(
    tmp_path, monkeypatch, package_indexes,
):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the real dependency-runtime test")

    server, handler = package_indexes
    wheel = _wheel("unhashed-sentinel", "1.0.0", {"unhashed_sentinel.py": b"VALUE = 1\n"})
    mirror_index = _index(server, handler, "mirror", {"unhashed-sentinel": wheel})
    for key in list(os.environ):
        if key.startswith(("PIP_", "UV_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PIP_INDEX_URL", mirror_index)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    from pm.environment import PythonEnvironment
    from pm.runtime import runtime_environment

    environment = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "venv",
        cache=tmp_path / "uv-cache", env=runtime_environment(), no_config=True,
    )
    environment.create()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("unhashed-sentinel==1.0.0\n", encoding="utf-8")
    with pytest.raises(InstallError, match="(?i)hash"):
        environment._install_requirements_file(requirements, require_hashes=True)

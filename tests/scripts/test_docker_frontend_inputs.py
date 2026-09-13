"""Exercise the frontend COPY closure before an image/network build.

PM environment construction is the only substituted boundary. The copied icon
provider must load, ask for its real dependency group, and invoke the copied
generator. Image-level runtime/permission coverage stays in tests/docker.
"""
from pathlib import Path
import runpy
import shlex
import shutil
from types import SimpleNamespace
from types import ModuleType

from pathspec import PathSpec

import pm


def test_frontend_copy_closure_reaches_the_icon_provider(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    # Apply the frontend's local COPY declarations, not a duplicated filename
    # allowlist. Runtime-base supplies PM itself and is separately image-tested.
    frontend = False
    ignored = PathSpec.from_lines("gitignore", (repo / ".dockerignore").read_text().splitlines())

    def excluded(directory, names):
        return [name for name in names if ignored.match_file(
            (Path(directory) / name).relative_to(repo).as_posix() +
            ("/" if (Path(directory) / name).is_dir() else ""))]
    for line in (repo / "Dockerfile").read_text().splitlines():
        if not line.startswith(("FROM ", "COPY ")):
            continue
        words = shlex.split(line.rstrip("\\"), comments=True)
        if not words:
            continue
        if words[0] == "FROM":
            frontend = words[-1] == "frontend_build"
        if not frontend or words[0] != "COPY" or any(word.startswith("--") for word in words[1:]):
            continue
        sources, destination = words[1:-1], tmp_path / words[-1]
        for pattern in sources:
            for source in repo.glob(pattern):
                target = destination / source.name if words[-1].endswith("/") else destination
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True,
                                    ignore=excluded)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    acquired = []

    def environment(**kwargs):
        acquired.append(kwargs)
        return Path("/prepared/icon-python")

    launched = []
    monkeypatch.setattr(pm, "build_environment", environment)
    provider = runpy.run_path(str(tmp_path / "scripts/build/icon_environment.py"))

    def run(argv, *, cwd):
        assert Path(argv[2]).is_file(), "generator must also be in the stage"
        assert cwd == tmp_path
        launched.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(provider["subprocess"], "run", run)
    assert provider["main"](["--source", str(tmp_path), "--out", str(tmp_path / "icons")]) == 0
    assert acquired[0]["source"] == tmp_path
    assert acquired[0]["groups"] == ["icon-build"]
    assert acquired[0]["only_groups"] is True
    assert launched[0][:2] == ["/prepared/icon-python", "-I"]

    # Exercise the generator's own source loader. Rasterization is not needed
    # to prove that every composed SVG's artwork made it into the build context.
    monkeypatch.setitem(__import__("sys").modules, "resvg_py", ModuleType("resvg_py"))
    generator = runpy.run_path(str(tmp_path / "scripts/generate_icons.py"))
    monkeypatch.setitem(generator["IconArt"].__init__.__globals__, "girl_bbox", lambda *args: (0, 0, 512, 512))
    art = generator["IconArt"](tmp_path)
    assert art.master_mac and art.master_mac_dark

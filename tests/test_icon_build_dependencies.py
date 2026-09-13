"""Icon tooling must not join the runtime's all-extras dependency set."""
import re
import shutil
import subprocess
from pathlib import Path


def test_icon_group_is_separate_from_runtime_extras():
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    assert uv, "uv is required for the dependency-selection contract"

    def packages(*selection):
        result = subprocess.run(
            [uv, "export", "--frozen", "--offline", "--no-hashes", "--no-annotate",
             "--no-header", "--no-emit-project", *selection],
            cwd=root, capture_output=True, text=True, encoding="utf-8", check=True,
        )
        return set(re.findall(r"^([a-z0-9-]+)==", result.stdout, re.MULTILINE))

    assert "resvg-py" not in packages("--all-extras")
    assert packages("--only-group", "icon-build") == {"pillow", "resvg-py"}

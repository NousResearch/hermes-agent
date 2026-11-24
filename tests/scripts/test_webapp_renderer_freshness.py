"""A Webapp renderer receipt is current only under the renderer environment it was built with.

Vite inlines ``VITE_*`` and ``NODE_ENV`` selects the build mode, so a development or perf-probe
bundle does not satisfy a production launch, and a production bundle does not satisfy either.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FRESHNESS = ROOT / "scripts" / "build" / "freshness.mjs"


def _env(**renderer: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "NODE_ENV" and not k.startswith("VITE_")}
    env.update(renderer)
    return env


def test_webapp_receipt_tracks_the_renderer_environment(tmp_path):
    node = shutil.which("node")
    assert node, "the freshness contract test requires the build's Node prerequisite"
    source = tmp_path / "source"
    (source / "apps" / "desktop").mkdir(parents=True)
    (source / "apps" / "desktop" / "main.ts").write_text("export {}\n", encoding="utf-8")
    out = tmp_path / "dist-webapp"
    out.mkdir()
    (out / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
    record = (
        f"import {{ buildInputs, recordProduct }} from {json.dumps(FRESHNESS.as_uri())};"
        f"const source = {json.dumps(str(source))};"
        f"recordProduct({{ source, product: 'webapp', out: {json.dumps(str(out))},"
        " inputs: buildInputs(source, 'webapp') })")

    def build(**renderer: str) -> None:
        subprocess.run([node, "--input-type=module", "-e", record], env=_env(**renderer), check=True)

    def current(**renderer: str) -> bool:
        ran = subprocess.run(
            [node, str(FRESHNESS), "--source", str(source), "--product", "webapp", "--out", str(out)],
            env=_env(**renderer), capture_output=True, text=True, check=True)
        return json.loads(ran.stdout)

    build(NODE_ENV="development", VITE_PERF_PROBE="1")
    assert current(NODE_ENV="development", VITE_PERF_PROBE="1") is True
    assert current() is False
    assert current(NODE_ENV="production") is False

    build()
    assert current() is True
    assert current(NODE_ENV="production") is True  # an unset NODE_ENV builds production
    assert current(VITE_PERF_PROBE="1") is False

"""The spectrum-ts patch that surfaces iMessage stream heartbeats (#124010).

spectrum-ts never passes ``onHeartbeat`` to advanced-imessage's
``createGrpcClient``; ``patch-spectrum-stream-heartbeat.mjs`` wraps that import
so heartbeats reach ``globalThis.__hermesPhotonStreamHeartbeat``. These tests
run the real patcher under node against a minimal fixture of the published
import line and execute the patched module.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

_PATCHER = Path("plugins/platforms/photon/sidecar/patch-spectrum-stream-heartbeat.mjs").resolve()
_IMPORT = (
    'import { ErrorCode, NotFoundError, ValidationError, createGrpcClient } '
    'from "@photon-ai/advanced-imessage/grpc";\n'
)
_BODY = "export const build = (options) => createGrpcClient(options);\n"


def _write_fixture(root: Path, source: str) -> Path:
    grpc = root / "node_modules" / "@photon-ai" / "advanced-imessage"
    grpc.mkdir(parents=True)
    (grpc / "package.json").write_text(
        json.dumps({"name": "@photon-ai/advanced-imessage", "type": "module", "exports": {"./grpc": "./grpc.js"}}),
        encoding="utf-8",
    )
    # Stand-in client: fires the heartbeat callback it was given, twice.
    (grpc / "grpc.js").write_text(
        "export const ErrorCode = {}, NotFoundError = Error, ValidationError = Error;\n"
        "export function createGrpcClient(options) { options.onHeartbeat?.(); options.onHeartbeat?.(); return options; }\n",
        encoding="utf-8",
    )
    dist = root / "node_modules" / "@spectrum-ts" / "imessage" / "dist"
    dist.mkdir(parents=True)
    target = dist / "index.js"
    target.write_text(source, encoding="utf-8")
    return target


def _node(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=cwd, text=True, capture_output=True, check=False,
    )


def test_patch_routes_heartbeats_to_sidecar_hook_and_is_idempotent(tmp_path: Path) -> None:
    target = _write_fixture(tmp_path, _IMPORT + _BODY)
    script = f"""
        import {{ patchSpectrumStreamHeartbeat }} from {json.dumps(_PATCHER.as_uri())};
        const first = patchSpectrumStreamHeartbeat({json.dumps(str(tmp_path))});
        const second = patchSpectrumStreamHeartbeat({json.dumps(str(tmp_path))});
        let hooked = 0, own = 0;
        globalThis.__hermesPhotonStreamHeartbeat = () => {{ hooked += 1; }};
        const mod = await import({json.dumps(target.as_uri())});
        mod.build({{ address: "x", onHeartbeat: () => {{ own += 1; }} }});
        process.stdout.write(JSON.stringify({{ first: first.patched, second: second.reason, hooked, own }}));
    """
    run = _node(script, tmp_path)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"first": True, "second": "already patched", "hooked": 2, "own": 2}


def test_patch_leaves_unknown_spectrum_shape_untouched(tmp_path: Path) -> None:
    """A reshaped import must not be rewritten; the CLI logs and exits 0 so
    ``npm ci`` still succeeds and the watchdog keeps its probe fallback."""
    original = _IMPORT.replace("createGrpcClient }", "createGrpcClient, createHttpClient }") + _BODY
    target = _write_fixture(tmp_path, original)
    run = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        text=True, capture_output=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "heartbeat patch skipped" in run.stderr
    assert target.read_text(encoding="utf-8") == original

"""Cold setup consumes lock values before executing any downloaded tool."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from pm.store import current_target


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.platforms("windows", "posix")
@pytest.mark.parametrize("missing_target", [False, True])
def test_cold_setup_uses_exact_lock_values(tmp_path, missing_target):
    checkout = tmp_path / "checkout"
    (checkout / "pm").mkdir(parents=True)
    shutil.copy2(ROOT / "setup-hermes.sh", checkout / "setup-hermes.sh")
    lock = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))
    target = current_target()
    uv_pin = lock["packages"]["uv"]
    artifact = uv_pin["artifacts"][target]
    if missing_target:
        del uv_pin["artifacts"][target]
    (checkout / "pm" / "lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    corrupt = tmp_path / "corrupt-download"
    corrupt.write_bytes(b"intentional digest mismatch; this must never be executed\n")
    args = tmp_path / "curl-args"
    hook = tmp_path / "transport.sh"
    hook.write_text('''curl() {
    printf '%s\\n' "$@" > "$PROBE_ARGS"
    while [ "$#" -gt 0 ]; do
        if [ "$1" = -o ]; then
            cp "$PROBE_DOWNLOAD" "$2"
            return
        fi
        shift
    done
    return 37
}
''', encoding="utf-8")
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_RUNTIME_DIR=(tmp_path / "tools").as_posix(), BASH_ENV=hook.as_posix(),
               PROBE_ARGS=args.as_posix(), PROBE_DOWNLOAD=corrupt.as_posix())
    result = subprocess.run(["bash", str(checkout / "setup-hermes.sh")], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    if missing_target:
        assert not args.exists(), "a missing target must not select a sibling artifact"
        assert f"no uv artifact for {target}" in result.stderr
    else:
        assert args.read_text(encoding="utf-8").splitlines()[-1] == artifact["url"]
        assert f"pinned uv {uv_pin['version']} ({target})" in result.stdout
        assert f"pinned {artifact['sha256']})" in result.stderr
    assert not (checkout / ".env").exists()

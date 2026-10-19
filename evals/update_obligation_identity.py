"""Disposable process/state-file A/B: run with the checkout path as argv[1].

No services are discovered or restarted. Both children write runtime status
through production code; only their temporary version fields are altered for
negative controls. Exit 1 on base, exit 0 when current successors settle.
"""

import json, os, pathlib, subprocess, sys, tempfile, time

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo))
root = pathlib.Path(tempfile.mkdtemp(prefix="obligations-live-"))
os.environ["HOME"] = str(root)
os.environ["HERMES_HOME"] = str(root / ".hermes")
from hermes_cli import update_cmd_fleet as fleet, update_receipt as receipts

print("MODULE", fleet.__file__)
sha = fleet._current_checkout_sha()
children = []
rows = []
try:
    for name in ["alpha", "beta"]:
        home = root / ".hermes" / "profiles" / name
        home.mkdir(parents=True)
        code = "import sys,time; sys.path.insert(0,sys.argv[1]); from gateway.status import write_runtime_status; write_runtime_status(gateway_state='running'); print('ready',flush=True); time.sleep(120)"
        p = subprocess.Popen(
            [sys.executable, "-c", code, str(repo)],
            cwd=repo,
            env={**os.environ, "HERMES_HOME": str(home)},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(p)
        assert p.stdout.readline().strip() == "ready", p.stderr.read()
    d = root / ".hermes/logs/update_receipts"
    d.mkdir(parents=True)
    receipt = {
        "outcome": "failed",
        "plan": {
            "runtimes": [
                {"kind": "gateway", "profile": name, "code_sha": "old", "pid": 1}
                for name in ["alpha", "beta"]
            ]
        },
    }
    (d / "latest.json").write_text(json.dumps(receipt))

    def observe(label):
        row = {
            "case": label,
            "fleet": receipts.collect_fleet_versions(),
            "pending": fleet._pending_fleet_restart_needed(),
        }
        rows.append(row)
        print(json.dumps(row))

    observe("current-successors")
    home = root / ".hermes/profiles/beta"
    state = home / "gateway_state.json"
    payload = json.loads(state.read_text())
    payload["code_sha"] = "old"
    state.write_text(json.dumps(payload))
    observe("stale-beta")
    payload["code_sha"] = None
    state.write_text(json.dumps(payload))
    observe("unknown-beta")
    children[1].terminate()
    children[1].wait(timeout=10)
    observe("missing-beta-with-current-alpha")
    assert rows[1]["pending"] and rows[2]["pending"] and rows[3]["pending"]
    print("VERDICT:", "FIXED" if not rows[0]["pending"] else "REPRODUCED")
    assert json.loads((d / "latest.json").read_text()) == receipt
    assert not rows[0]["pending"], (
        "Current identity-matched successors must settle the warning"
    )
finally:
    for p in children:
        if p.poll() is None:
            p.terminate()
        p.wait(timeout=10)
    print("CLEANUP", [p.returncode for p in children])

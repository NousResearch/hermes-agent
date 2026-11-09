import json, sys, time
from hermes_cli.plugins import discover_plugins, invoke_hook
discover_plugins(force=True)
mod = sys.modules["hermes_plugins.model_doctor.model_doctor"]

# 0) start from a clean slate — the snapshot is real operator state, not test state
mod._snapshot_path().unlink(missing_ok=True)
mod.RECORDS.clear()
mod._load_snapshot()


# 1) hook-driven recording through the REAL loader
invoke_hook("post_api_request", model="live/probe", provider="p",
            base_url="u", status_code=200, finish_reason="stop",
            assistant_content_chars=50, api_duration=0.5)
invoke_hook("api_request_error", model="live/probe", provider="p",
            base_url="u", status_code=401,
            error={"type": "auth", "message": "bad key"})

st = mod.tool_status({})
row = [r for r in st["models"] if r["model"] == "live/probe"][0]
assert row["counts"]["ok"] == 1 and row["counts"]["auth"] == 1, row
print("PASS hooks record events:", row["counts"])

# 2) classification correctness
c = mod._classify(401, "", 0, 0, {"type": "auth", "message": "x"})
assert c["verdict"] == "AUTH", c
c = mod._classify(503, "", 0, 1, None)
assert c["verdict"] == "UPSTREAM", c
c = mod._classify(200, "length", 0, 1, None)
assert c["verdict"] == "EMPTY", c
c = mod._classify(None, "", 0, 0, {"type": "timeout", "message": "t"})
assert c["verdict"] == "ERROR", c
c = mod._classify(200, "stop", 100, 2, None)
assert c["verdict"] == "OK", c
print("PASS classification: AUTH/UPSTREAM/EMPTY/ERROR/OK")

# 3) snapshot roundtrip
mod._save_snapshot()
before = len(mod.RECORDS)
mod.RECORDS.clear()
mod._load_snapshot()
assert len(mod.RECORDS) == before, "snapshot lost records"
print("PASS snapshot roundtrip:", before, "records")

print("ALL CHECKS PASSED")

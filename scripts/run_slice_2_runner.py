import subprocess, sys, json, time
with open("/tmp/slices.json") as f:
    data = json.load(f)
files = data["slice"][1]["files"]
if isinstance(files, str):
    files = files.split(":")
files = [f.strip() for f in files if f.strip()]
print(f"SLICE 2: running {len(files)} test files (timeout 3600s)")
pytest_path = "/Users/mikedemott/.hermes/worktrees/hermes-pr-116/.venv/bin/pytest"
cmd = [pytest_path, "-x", "-q", "--tb=short"] + files
res = subprocess.run(cmd, stdout=open("/tmp/slice2_stdout_fixed.log", "w"), stderr=subprocess.STDOUT)
with open("/tmp/slice2_result_fixed.txt", "w") as out:
    out.write(str(res.returncode) + "\n")
print(f"Done, exit code {res.returncode}")

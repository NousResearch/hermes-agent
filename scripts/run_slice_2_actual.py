import subprocess, sys, time
with open("/tmp/slice_2_files.txt") as f: files = [line.strip() for line in f if line.strip()]
print(f"SLICE 2 ACTUAL: {len(files)} files")
pytest_path = "/Users/mikedemott/.hermes/worktrees/hermes-pr-116/.venv/bin/pytest"
cmd = [pytest_path, "-x", "-q", "--tb=short"] + files
res = subprocess.run(cmd, stdout=open("/tmp/slice2_actual_stdout.log","w"), stderr=subprocess.STDOUT)
with open("/tmp/slice2_actual_exitcode.txt","w") as f: f.write(str(res.returncode)+"\n")
print(f"Slice 2 done, exit={res.returncode}, time={time.time()}")

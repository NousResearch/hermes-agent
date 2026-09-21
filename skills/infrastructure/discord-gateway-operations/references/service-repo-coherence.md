# Service-Repo Coherence

A common source of "the feature is in the code but doesn't work" failures in the Discord gateway setup.

## Setup

The live gateway services on Sahil's VPS do NOT run from the git repository checkout at `/home/kensei/repos/KenseiAgent/`. They run from the Hermes package directory at `/home/kensei/.hermes/hermes-agent/` which is populated during installation/update, NOT on every git push.

This creates two copies of the same code:
- **Repo checkout**: `/home/kensei/repos/KenseiAgent/` — where development happens, where `git pull`, `git diff`, `git log` work.
- **Service runtime**: `/home/kensei/.hermes/hermes-agent/` — where systemd ExecStart points, where the actual Python interpreter loads modules from.

## The failure mode

```
1. Feature developed in repo checkout (e.g. auto-join in gateway/run.py)
2. Code committed, PR merged
3. Service NOT updated/reinstalled → runtime still has old code
4. Bot restarts → loads old code → feature "doesn't work"
5. Developer checks repo checkout → code IS there → confusion
```

## How to verify coherence

Always ask: "Which copy is the service actually loading from?"

```bash
# 1. Check where the service ExecStart points
systemctl cat hermes-gateway-misa-misa | grep ExecStart
# → ExecStart=/home/kensei/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run

# 2. Verify the runtime file content
tail -20 /home/kensei/.hermes/hermes-agent/gateway/run.py
# vs
# Repo checkout
tail -20 /home/kensei/repos/KenseiAgent/gateway/run.py

# 3. The definitive check: grep for the feature keyword in BOTH copies
# Example: auto_join
for p in /home/kensei/.hermes/hermes-agent /home/kensei/repos/KenseiAgent; do
  echo "=== $p ==="
  grep -c "auto_join" "$p/plugins/platforms/discord/adapter.py"
done
```

If the counts differ, coherence is broken. The service is running stale code.

## Path substitution

When editing files via `patch` or `write_file`, the tool writes to the path you specify. If you specify `/home/kensei/repos/KenseiAgent/gateway/run.py`, ONLY the repo is updated. The service continues to run old code.

**After any edit to repo checkout code that MUST be live, sync to the service directory.**

### Sync methods

#### Method A: pip install from repo (recommended for permanent fixes)

```bash
cd /home/kensei/repos/KenseiAgent
source /home/kensei/.hermes/hermes-agent/venv/bin/activate
pip install -e .
```

This updates the installed package so the service starts with new code.

#### Method B: manual rsync (quick fix, not persistent)

```bash
# Dangerous — do NOT use unless you know what you're doing
rsync -av --exclude='.git' /home/kensei/repos/KenseiAgent/ /home/kensei/.hermes/hermes-agent/
```

This overwrites the service directory with repo content. Will be lost on next `pip install`.

#### Method C: runtime hot-patch (emergency only)

```bash
# Directly edit the service file
sudo sed -i 's/old_string/new_string/' /home/kensei/.hermes/hermes-agent/gateway/run.py
# Restart the service
sudo systemctl restart hermes-gateway-<profile>
```

This works for one-line fixes but creates divergence. Always follow up with Method A.

#### Method D: change systemd to point at repo (not recommended)

```bash
# Do NOT do this — it creates circular dependency issues
# and makes updates via normal pip install impossible
```

## Prevention

1. **After every repo edit that affects gateway behavior, run `pip install -e .`** before claiming the feature works.
2. **After every systemctl restart, verify the restarted process shows the expected log line.** Check for the feature keyword in the gateway log.
3. **Document in skills and runbooks that the service directory is canonical, not the repo checkout.**

## Pitfalls

- `hermes cli version` reports the package version, NOT the git commit. A version bump does NOT guarantee the code on disk is that version.
- `git status` in the repo checkout tells you nothing about the service directory.
- `systemctl restart` restarts the service but does NOT re-install the package. If you edited the repo and NOT the service directory, the restart is meaningless.
- The `venv/bin` path in ExecStart is the Python in the VENV, but the module path (`-m hermes_cli.main`) resolves to wherever `hermes_agent` package is installed in that venv. Usually that's `.hermes/hermes-agent/`, not the repo.

## Verification script

```bash
#!/bin/bash
# verify-coherence.sh — compare repo vs service for voice feature keywords
set -euo pipefail

KEYWORDS=("auto_join" "voice_channel" "VoiceReceiver" "handle_voice_state")
REPO="/home/kensei/repos/KenseiAgent"
SRV="/home/kensei/.hermes/hermes-agent"
FILE="plugins/platforms/discord/adapter.py"

for kw in "${KEYWORDS[@]}"; do
  repo_count=$(grep -c "$kw" "$REPO/$FILE" || true)
  srv_count=$(grep -c "$kw" "$SRV/$FILE" || true)
  if [ "$repo_count" != "$srv_count" ]; then
    echo "DIVERGENCE on '$kw': repo=$repo_count service=$srv_count"
  fi
done
echo "Coherence check complete."
```

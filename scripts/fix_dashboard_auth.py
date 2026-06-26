#!/usr/bin/env python3
"""Fix the dashboard basic_auth password hash in data/config.yaml.

Usage (from host, with containers running):
    docker exec hermes-web python3 /opt/data/fix_dashboard_auth.py

Or copy into the container and run:
    docker cp scripts/fix_dashboard_auth.py hermes-web:/tmp/fix_auth.py
    docker exec hermes-web python3 /tmp/fix_auth.py
"""
import yaml

HASH = "scrypt$16384$8$1$QJMs7wfRIelUP6tXKnzgTg==$d4nrKch8xCpKe5pV5hGK5S+CnJrLLF8idO56e58Tlhw="

with open("/opt/data/config.yaml") as f:
    c = yaml.safe_load(f)

c.setdefault("dashboard", {})
c["dashboard"]["basic_auth"] = {
    "username": "admin",
    "password_hash": HASH,
}

with open("/opt/data/config.yaml", "w") as f:
    yaml.dump(c, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

# Verify
with open("/opt/data/config.yaml") as f:
    c2 = yaml.safe_load(f)
stored = c2["dashboard"]["basic_auth"]["password_hash"]
print("Stored hash:", stored)
print("Match:", stored == HASH)
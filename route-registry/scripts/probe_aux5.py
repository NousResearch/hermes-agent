#!/usr/bin/env python3
"""Reviewed probe 5: fingerprint identity — which credential does the ceecee
custom:commandcode client actually carry? Compares against known pool entries.
Never prints key material."""
import os
import sys


def fp(val):
    if not val:
        return None
    import hashlib
    return "len=%d sha=%s" % (len(val), hashlib.sha256(val.encode()).hexdigest()[:8])


def main():
    sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
    from hermes_cli.auth import read_credential_pool
    from agent import auxiliary_client as ac

    print("HERMES_HOME:", os.environ.get("HERMES_HOME"))
    entries = read_credential_pool("custom:commandcode")
    print("pool entries visible:", len(entries))
    for e in entries:
        print("  pool entry fp:", fp(str(e.get("access_token") or "")), "label:", e.get("label"), "source:", e.get("source"))

    client, model = ac.resolve_provider_client(
        "custom:commandcode", model="deepseek/deepseek-v4-flash",
        explicit_base_url="https://api.commandcode.ai/provider/v1",
    )
    if client:
        k = getattr(client, "api_key", "")
        print("client key fp:", fp(str(k or "")))
    else:
        print("client: None")


if __name__ == "__main__":
    main()
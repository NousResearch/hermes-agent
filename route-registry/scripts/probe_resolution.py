#!/usr/bin/env python3
"""Reviewed probe: how does a profile-gateway runtime resolve custom:* providers?

Run with HERMES_HOME pointed at a profile. Prints resolution outcomes only —
never key material (fingerprints only).
"""
import sys


def fp(val):
    if not val:
        return None
    import hashlib
    return "len=%d sha=%s" % (len(val), hashlib.sha256(val.encode()).hexdigest()[:8])


def main():
    sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
    from hermes_cli.runtime_provider import (
        _try_resolve_from_custom_pool,
        _get_named_custom_provider,
        resolve_requested_provider,
    )

    print("HERMES_HOME:", __import__("os").environ.get("HERMES_HOME"))

    # 1. Named custom provider resolution (config path)
    for name in ("custom:commandcode", "custom:CommandCode", "custom:turbohaul-local"):
        r = _get_named_custom_provider(name)
        if r:
            print(f"_get_named_custom_provider({name}) -> provider={r.get('provider')} base_url={r.get('base_url')} key={fp(r.get('api_key'))}")
        else:
            print(f"_get_named_custom_provider({name}) -> None")

    # 2. Pool-by-base_url resolution
    for url in (
        "https://api.commandcode.ai/provider/v1",
        "http://127.0.0.1:11410/v1",
        "https://api.xkiro.com/v1",
        "https://api.b.ai/v1",
    ):
        r = _try_resolve_from_custom_pool(url, "custom-probe")
        if r:
            print(f"pool({url}) -> source={r.get('source')} key={fp(r.get('api_key'))}")
        else:
            print(f"pool({url}) -> None")

    # 3. Full resolve of an arbitrary provider request
    from hermes_cli.runtime_provider import resolve_runtime_provider
    for prov in ("custom:commandcode", "custom:turbohaul-local"):
        try:
            r = resolve_runtime_provider(requested=prov)
            print(f"resolve_runtime_provider({prov}) -> provider={r.get('provider')} api_mode={r.get('api_mode')} base_url={r.get('base_url')} key={fp(r.get('api_key'))} source={r.get('source')}")
        except Exception as e:
            print(f"resolve_runtime_provider({prov}) -> ERROR {type(e).__name__}: {e}")

    # 4. What does read_credential_pool see (merged profile+global)?
    from hermes_cli.auth import read_credential_pool
    merged = read_credential_pool(None)
    print("merged pool keys:", sorted(k for k in merged if isinstance(merged.get(k), list) and merged.get(k)))


if __name__ == "__main__":
    main()
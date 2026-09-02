#!/usr/bin/env python3
"""Reviewed probe: aux-router resolution of named custom providers in profile context.

Mirrors what try_activate_fallback does: resolve_provider_client(fb_provider,
model, explicit_base_url=fb.base_url, explicit_api_key=resolve_entry_api_key(fb)).
Never prints key material — fingerprints only.
"""
import os
import sys


def fp(val):
    if not val:
        return None
    import hashlib
    return "len=%d sha=%s" % (len(val), hashlib.sha256(val.encode()).hexdigest()[:8])


def main():
    sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.fallback_config import resolve_entry_api_key

    print("HERMES_HOME:", os.environ.get("HERMES_HOME"))
    cases = [
        # (provider, model, base_url) — mimic emitted fallback entries
        ("custom:commandcode", "deepseek/deepseek-v4-flash", "https://api.commandcode.ai/provider/v1"),
        ("custom:bai", "glm-5.3-flash", "https://api.b.ai/v1"),
        ("custom:xkiro-pro", "deepseek/deepseek-v4-pro", "https://api.xkiro.com/v1"),
    ]
    for prov, model, base in cases:
        entry = {"provider": prov, "model": model, "base_url": base}
        hint = resolve_entry_api_key(entry)
        client, resolved = resolve_provider_client(
            prov, model=model, raw_codex=True,
            explicit_base_url=base, explicit_api_key=hint,
        )
        if client is None:
            print(f"{prov}: FAILED to resolve client")
            continue
        cb = str(getattr(client, "base_url", "") or "")
        # try to find the api key the client was built with
        key = getattr(client, "api_key", None)
        if callable(key) and not isinstance(key, str):
            key = "<callable>"
        print(f"{prov}: client OK base={cb} key={fp(str(key or ''))} model={resolved}")
        # what does the client's default_headers say (names only)?
        hdrs = getattr(client, "default_headers", None)
        if isinstance(hdrs, dict):
            print(f"    default_headers keys: {sorted(hdrs.keys())}")


if __name__ == "__main__":
    main()
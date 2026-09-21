#!/usr/bin/env python3
"""Reviewed probe 2: trace why named custom provider resolution returns None."""
import os
import sys
import traceback


def main():
    sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
    from hermes_cli.runtime_provider import _get_named_custom_provider

    print("HERMES_HOME:", os.environ.get("HERMES_HOME"))
    entry = _get_named_custom_provider("custom:commandcode")
    print("named entry:", entry if entry else None)

    from agent import auxiliary_client as ac
    from hermes_cli.runtime_provider import resolve_runtime_provider

    # resolve_runtime_provider path (main-agent primary path)
    try:
        r = resolve_runtime_provider(requested="custom:commandcode")
        print("resolve_runtime_provider:", {k: v for k, v in r.items() if k not in ("credential_pool",)})
    except Exception as e:
        print("resolve_runtime_provider ERROR:", type(e).__name__, e)

    # aux router with raw_codex=False
    try:
        client, model = ac.resolve_provider_client(
            "custom:commandcode", model="deepseek/deepseek-v4-flash",
            explicit_base_url="https://api.commandcode.ai/provider/v1",
        )
        print("aux client:", client, "model:", model)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
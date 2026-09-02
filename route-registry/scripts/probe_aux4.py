#!/usr/bin/env python3
"""Reviewed probe 4: trace the named-custom branch return path."""
import os
import sys
import traceback


def main():
    sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
    from agent import auxiliary_client as ac

    print("HERMES_HOME:", os.environ.get("HERMES_HOME"))
    try:
        client, model = ac.resolve_provider_client(
            "custom:commandcode",
            model="deepseek/deepseek-v4-flash",
            explicit_base_url="https://api.commandcode.ai/provider/v1",
            explicit_api_key=None,
        )
        print("RESULT client:", client, "model:", model)
    except Exception:
        traceback.print_exc()

    # Also with api key hint set (simulating key_env resolution upstream)
    try:
        client, model = ac.resolve_provider_client(
            "custom:commandcode",
            model="deepseek/deepseek-v4-flash",
            explicit_base_url="https://api.commandcode.ai/provider/v1",
            explicit_api_key="sk-test-fake-not-real",
        )
        print("RESULT with hint: client:", type(client).__name__ if client else None, "model:", model)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Reviewed probe 3: step through the named-custom branch conditions."""
import os
import sys


def main():
    sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
    from agent.auxiliary_client import _normalize_aux_provider
    from hermes_cli.runtime_provider import _get_named_custom_provider

    print("HERMES_HOME:", os.environ.get("HERMES_HOME"))
    for raw in ("custom:commandcode", "custom:bai", "custom:xkiro-pro"):
        norm = _normalize_aux_provider(raw)
        e1 = _get_named_custom_provider(raw)
        e2 = _get_named_custom_provider(norm)
        print(f"raw={raw!r} -> normalized={norm!r}")
        print(f"   entry(raw)={bool(e1)} entry(norm)={bool(e2)}")
        if e1:
            print(f"   base_url={e1.get('base_url')!r} api_key_set={bool((e1.get('api_key') or '').strip())} key_env={e1.get('key_env')!r}")


if __name__ == "__main__":
    main()
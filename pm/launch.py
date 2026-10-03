"""CLI entry after the isolated interpreter has been selected."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> int:
    # PM's import closure constructs HTTPS clients; install platform trust
    # before importing it, but never mutate SSL merely by importing launch.
    # Route through the shared guard: a raw inject here can double-inject when
    # another code path already ran the guarded install (see #126808).
    from agent.ssl_verify import install_truststore

    install_truststore()
    from pm.cli import main as cli_main
    from pm.runtime import lease_current_runtime

    lease_current_runtime()
    return cli_main()

if __name__ == "__main__":
    raise SystemExit(main())

"""Allow ``python -m hermes_office`` to launch the office UI."""
from .launcher import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

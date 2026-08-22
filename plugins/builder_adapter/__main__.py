"""Explicit module entrypoint; importing the package remains inert."""

from .runtime import main

raise SystemExit(main())

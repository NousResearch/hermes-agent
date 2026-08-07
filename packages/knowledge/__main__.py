"""Allow `python -m packages.knowledge` to run the sync worker."""
from .worker import main

raise SystemExit(main())

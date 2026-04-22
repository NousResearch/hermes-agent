"""
Hermes CLI - Unified command-line interface for Hermes Agent.

Provides subcommands for:
- hermes chat          - Interactive chat (same as ./hermes)
- hermes gateway       - Run gateway in foreground
- hermes gateway start - Start gateway service
- hermes gateway stop  - Stop gateway service  
- hermes setup         - Interactive setup wizard
- hermes status        - Show status of all components
- hermes cron          - Manage cron jobs
"""

from .version import __release_date__, __version__

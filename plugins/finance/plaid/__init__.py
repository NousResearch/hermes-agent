"""Plaid backend for the finance plugin.

Plaid is the reference :class:`~plugins.finance.provider.FinanceProvider`. It
talks to the Plaid REST API directly over the core ``httpx`` dependency (no
``plaid-python`` SDK) to keep the supply-chain footprint at zero new packages.
"""

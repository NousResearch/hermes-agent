"""
Nomad integration for atropos-agent.

Provides:
- NomadClient: Client for Nomad HTTP API
- Job templates for sandbox containers
"""

from .client import NomadClient

__all__ = ["NomadClient"]

"""SPC-CTX context engine plugin for Hermes Agent.

Bridge to the hermes_spc_plugin package installed from
/Users/blitz/dev_ops/self-evolving/context-engine-py/src.
"""
from hermes_spc_plugin import SPCContextEngine, load_engine, register

__all__ = ["SPCContextEngine", "load_engine", "register"]

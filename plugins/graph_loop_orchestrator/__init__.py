"""Graph-and-Loop Hermes plugin.

Provides persistent master/worker/verifier/router tools to every enabled
Hermes profile without exposing credentials or mutating conversation prompts.
"""
from .tools import register_tools

def register(ctx):
    register_tools(ctx)

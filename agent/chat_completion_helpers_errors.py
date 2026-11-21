"""Streaming exception types that must not lazy-load a provider SDK."""
import sys


def anthropic_connection_error_types() -> tuple:
    # An error instance implies the SDK is already imported; error recovery must
    # not import (or lazy-install) an otherwise unused SDK.
    anthropic = sys.modules.get("anthropic")
    return (anthropic.APIConnectionError,) if anthropic is not None else ()

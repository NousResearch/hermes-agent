"""Shared helpers for W4 / F-013 tool-call-parser tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def args_as_dict(tc) -> dict:
    """Decode a ChatCompletionMessageToolCall's arguments JSON string."""
    return json.loads(tc.function.arguments)

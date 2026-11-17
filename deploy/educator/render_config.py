"""Render config.yaml.template with a values file.

    python deploy/educator/render_config.py values.yaml > config.yaml

``values.yaml`` maps each placeholder name (without braces) to its value,
e.g. ``HOME_CHANNEL_ID: C0123ABCD``. Every placeholder must be filled and
every value must be used; the result must parse as YAML. Values are inserted
as text inside the template's quotes, so they must not contain a double
quote or a newline.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parent / "config.yaml.template"
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def render(template: str, values: dict) -> str:
    names = placeholders(template)
    missing = sorted(names - set(values))
    unused = sorted(set(values) - names)
    if missing or unused:
        raise ValueError(f"missing values: {missing}; unknown values: {unused}")
    for name, value in values.items():
        text = "" if value is None else str(value)
        if '"' in text or "\n" in text:
            raise ValueError(f"{name}: value may not contain a double quote or newline")
    out = PLACEHOLDER.sub(lambda m: str(values[m.group(1)] or ""), template)
    yaml.safe_load(out)  # must still parse
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    values = yaml.safe_load(Path(argv[1]).read_text(encoding="utf-8")) or {}
    try:
        sys.stdout.write(render(TEMPLATE.read_text(encoding="utf-8"), values))
    except ValueError as e:
        print(f"render_config: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

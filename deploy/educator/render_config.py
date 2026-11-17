"""Fill the educator package's {{…}} placeholders from one values file.

    python deploy/educator/render_config.py values.yaml                 > config.yaml
    python deploy/educator/render_config.py values.yaml SOUL.md         > SOUL.md
    python deploy/educator/render_config.py values.yaml slack-app-manifest.yaml

``values.yaml`` maps each placeholder name (without braces) to its value,
e.g. ``HOME_CHANNEL_ID: C0123ABCD`` — start from values.example.yaml. Every
placeholder of the chosen file must be filled, and every key must be a
placeholder somewhere in the package (a typo is an error, not a no-op).
YAML output must still parse. Values are inserted as text inside the
template's quotes, so they may not contain a double quote or a newline.
``cron-jobs.yaml`` is filled by apply_cron_jobs.py from the same file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "config.yaml.template"
PACKAGE_FILES = ("config.yaml.template", "SOUL.md", "slack-app-manifest.yaml", "cron-jobs.yaml")
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def package_placeholders() -> set[str]:
    out: set[str] = set()
    for name in PACKAGE_FILES:
        out |= placeholders((HERE / name).read_text(encoding="utf-8"))
    return out


def render(template: str, values: dict, *, is_yaml: bool = True) -> str:
    missing = sorted(placeholders(template) - set(values))
    unknown = sorted(set(values) - package_placeholders())
    if missing or unknown:
        raise ValueError(f"missing values: {missing}; unknown values: {unknown}")
    for name, value in values.items():
        text = "" if value is None else str(value)
        if '"' in text or "\n" in text:
            raise ValueError(f"{name}: value may not contain a double quote or newline")
    out = PLACEHOLDER.sub(lambda m: "" if values[m.group(1)] is None else str(values[m.group(1)]),
                          template)
    if is_yaml:
        yaml.safe_load(out)  # must still parse
    return out


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    values = yaml.safe_load(Path(argv[1]).read_text(encoding="utf-8")) or {}
    path = HERE / argv[2] if len(argv) == 3 else TEMPLATE
    try:
        sys.stdout.write(render(path.read_text(encoding="utf-8"), values,
                                is_yaml=path.suffix in (".yaml", ".template")))
    except ValueError as e:
        print(f"render_config: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

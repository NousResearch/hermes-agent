import re
from typing import Any

from pip._internal.utils.compat import tomllib

REGEX = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"


class PEP723Exception(ValueError):
    """Raised to indicate a problem when parsing PEP 723 metadata from a script"""

    def __init__(self, msg: str) -> None:
        self.msg = msg


def pep723_metadata(scriptfile: str) -> dict[str, Any]:
    with open(scriptfile, encoding="utf8") as f:
        script = f.read()

    name = "script"
    matches = list(
        filter(lambda m: m.group("type") == name, re.finditer(REGEX, script))
    )

    if len(matches) > 1:
        raise PEP723Exception(f"Multiple {name!r} blocks found in {scriptfile!r}")
    elif len(matches) == 1:
        content = "".join(
            line[2:] if line.startswith("# ") else line[1:]
            for line in matches[0].group("content").splitlines(keepends=True)
        )
        try:
            metadata = tomllib.loads(content)
        except Exception as exc:
            raise PEP723Exception(f"Failed to parse TOML in {scriptfile!r}") from exc
    else:
        raise PEP723Exception(
            f"File does not contain {name!r} metadata: {scriptfile!r}"
        )

    return metadata

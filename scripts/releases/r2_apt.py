"""Live Termux APT references, independent of the desktop canary head."""
from collections.abc import Callable
import re


CANARY_ROOT = "releases/termux/canary/"
_INDEX = re.compile(r"releases/termux/canary/dists/[^/]+/[^/]+/binary-[^/]+/Packages(?:\.gz)?")


def package_filename(stanza: str) -> str:
    """Require an unambiguous deb822 record before trusting its pool reference."""
    from .r2 import relative_artifact_path

    fields: dict[str, str] = {}
    previous = ""
    for line in stanza.splitlines():
        if line.startswith((" ", "\t")) and previous not in ("", "package", "filename"):
            continue  # Description and other multiline fields do not identify the artifact.
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9-]*):[ \t]*(.*)", line)
        if match is None or match[1].lower() in fields:
            raise ValueError("Malformed or duplicate APT field")
        previous = match[1].lower()
        fields[previous] = match[2]
    filename = relative_artifact_path(fields.get("filename", ""))
    if not fields.get("package") or not filename.startswith("pool/") or not filename.endswith(".deb"):
        raise ValueError("APT record must name a package and a pool .deb")
    return filename


def live_pool_keys(keys: list[str], read: Callable[[str], str | None]) -> set[str]:
    """Read the uncompressed index the APT publisher emits beside Packages.gz."""
    indices = {key.removesuffix(".gz") for key in keys if _INDEX.fullmatch(key)}
    if not indices:
        # A staged pool with no readable repository is not evidence that its packages are dead.
        return {key for key in keys if key.startswith(CANARY_ROOT + "pool/")}
    protected = set()
    for key in sorted(indices):
        text = read(key)
        try:
            if text is None or not text.strip():
                raise ValueError("Missing APT index")
            stanzas = re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").strip())
            protected.update(CANARY_ROOT + package_filename(stanza) for stanza in stanzas)
        except ValueError as exc:
            raise RuntimeError(f"Cannot establish APT references from {key}; refusing to prune") from exc
    return protected

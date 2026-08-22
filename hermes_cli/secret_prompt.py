"""Secret input prompts with masked typing feedback."""

from __future__ import annotations

import getpass
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


_BACKSPACE_CHARS = {"\b", "\x7f"}
_ENTER_CHARS = {"\r", "\n"}
_EOF_CHARS = {"\x04", "\x1a"}


# A non-TTY setup or token rotation may be launched by a provider that injects
# a new value into the parent environment before Hermes starts.  The normal
# dotenv contract intentionally lets ``~/.hermes/.env`` override shell values,
# so retain only the configured provider token variable at this boundary before
# dotenv loading begins.  Never snapshot the whole environment.
_ROTATION_INPUTS: dict[str, str] = {}
_ROTATION_PROVIDERS = {
    "bitwarden": ("BWS_ACCESS_TOKEN", "access_token_env"),
    "onepassword": ("OP_SERVICE_ACCOUNT_TOKEN", "service_account_token_env"),
    "op": ("OP_SERVICE_ACCOUNT_TOKEN", "service_account_token_env"),
    "1password": ("OP_SERVICE_ACCOUNT_TOKEN", "service_account_token_env"),
}
_TOKEN_NAME_ENV_REF_RE = re.compile(r"\${([^}]+)}")
_ROTATION_EXPANSION_MAX_DEPTH = 16
# Keep dotenv-name traversal on the same finite budget as token-name
# expansion.  The files are persisted input and may be malformed or hostile;
# this limit is deliberately small enough that the pre-dotenv startup seam
# cannot become an unbounded resolver.
_ROTATION_CAPTURE_MAX_NODES = _ROTATION_EXPANSION_MAX_DEPTH


def _expand_token_name(value: str, source: Mapping[str, str]) -> str:
    """Resolve config-supported environment references in a token-name leaf.

    This mirrors config.yaml's ``${VAR}`` and ``${env:VAR}`` forms while
    leaving unresolved and non-environment SecretRef forms literal.  It reads
    only variables referenced by this leaf; the caller still snapshots only
    the resulting provider token names.
    """

    def resolve(current: str, stack: frozenset[str], depth: int) -> str:
        if depth >= _ROTATION_EXPANSION_MAX_DEPTH:
            return current

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            inner = match.group(1).strip()
            if inner.startswith("env:"):
                name = inner[len("env:"):].strip()
            elif ":" in inner and re.match(r"^[a-z][a-z0-9_-]*:", inner):
                return raw
            else:
                name = inner
            if not name or name in stack or name not in source:
                return raw
            return resolve(source[name], stack | {name}, depth + 1)

        return _TOKEN_NAME_ENV_REF_RE.sub(replace, current)

    return resolve(value, frozenset(), 0)


def _token_name_ref_names(value: str) -> set[str]:
    """Return only environment names referenced by one token-name leaf."""
    names: set[str] = set()
    for match in _TOKEN_NAME_ENV_REF_RE.finditer(value):
        inner = match.group(1).strip()
        if inner.startswith("env:"):
            name = inner[len("env:"):].strip()
        elif ":" in inner and re.match(r"^[a-z][a-z0-9_-]*:", inner):
            continue
        else:
            name = inner
        if name:
            names.add(name)
    return names


def _parse_rotation_dotenv_value(raw_value: str) -> str:
    """Parse a selected persisted dotenv value without broadening the read.

    Keep the line selection narrow, but match the canonical Hermes dotenv
    readers for the value syntax: quoted values may contain ``#`` and a
    trailing comment begins only after the closing quote; unquoted comments
    begin at a whitespace-delimited ``#``.  The lazy imports avoid a module
    cycle during the early config import while reusing the canonical escape
    handling whenever this function is reached after bootstrap.
    """
    value = raw_value.strip()
    if not value:
        return value

    try:
        from agent.secret_scope import _strip_inline_comment
        from hermes_cli.config import _parse_env_value

        return _parse_env_value(_strip_inline_comment(value))
    except Exception:  # noqa: BLE001 — capture must never block startup
        # Keep a local fail-open parser for unusual partial-import contexts.
        quote = value[0]
        if quote in {"'", '"'}:
            index = 1
            while index < len(value):
                char = value[index]
                if quote == '"' and char == "\\":
                    index += 2
                    continue
                if char == quote:
                    remainder = value[index + 1:].lstrip()
                    if remainder.startswith("#"):
                        value = value[: index + 1]
                    break
                index += 1
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()

        if len(value) >= 2 and value[0] == value[-1] == '"':
            quoted = value[1:-1]
            parsed: list[str] = []
            index = 0
            while index < len(quoted):
                char = quoted[index]
                if char == "\\" and index + 1 < len(quoted):
                    next_char = quoted[index + 1]
                    if next_char in {'"', "\\"}:
                        parsed.append(next_char)
                        index += 2
                        continue
                parsed.append(char)
                index += 1
            return "".join(parsed)
        if len(value) >= 2 and value[0] == value[-1] == "'":
            return value[1:-1]
        return value


def _read_rotation_dotenv_names(
    path: os.PathLike[str] | str,
    names: set[str],
) -> dict[str, str] | None:
    """Read only selected assignments from one persisted dotenv file.

    This deliberately does not parse or retain the file's unrelated values.
    Any read/decode failure returns no values so the pre-dotenv seam never
    broadens into an untrusted environment snapshot.
    """
    if not names:
        return {}
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return {}

    selected: dict[str, str] = {}
    pending = set(names)
    visited: set[str] = set()
    traversed = 0
    # Split once.  Each pass still examines only assignments whose key is in
    # the current narrow ``wanted`` set; unrelated values are never parsed or
    # retained.  The finite pass count is important because a reachable chain
    # otherwise causes a full-file rescan for every assignment.
    lines = text.splitlines()
    while pending:
        wanted = pending - visited
        if not wanted:
            break
        if traversed + len(wanted) > _ROTATION_CAPTURE_MAX_NODES:
            # ``None`` distinguishes an over-limit traversal from an empty
            # file.  The caller treats it as fail-closed and does not use a
            # partial snapshot from this source.
            return None
        traversed += len(wanted)
        visited.update(wanted)
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            if key in wanted:
                selected[key] = _parse_rotation_dotenv_value(raw_value)
        pending = {
            ref
            for key, value in selected.items()
            if key in wanted
            for ref in _token_name_ref_names(value)
            if ref not in visited
        }
    return selected


def _rotation_capture_source(
    raw_names: set[str],
    *,
    environ: Mapping[str, str],
    dotenv_sources: Sequence[tuple[os.PathLike[str] | str, bool]],
) -> dict[str, str]:
    """Resolve only token-name references against pre-dotenv inputs.

    ``dotenv_sources`` follows the effective loader order and pairs each file
    with its ``override`` behavior.  Only names referenced by a provider
    token-name leaf are read from those files; provider token values themselves
    continue to come solely from the pre-dotenv ``environ`` mapping.
    """
    referenced = {
        name
        for raw_name in raw_names
        for name in _token_name_ref_names(raw_name)
    }
    if not referenced:
        return {}
    if len(referenced) > _ROTATION_CAPTURE_MAX_NODES:
        return {}

    source = {
        name: environ[name]
        for name in referenced
        if name in environ
    }
    pending = set(referenced)
    for path, override in dotenv_sources:
        values = _read_rotation_dotenv_names(path, pending)
        if values is None:
            # A malformed/overly deep selected-name graph must not result in a
            # partial environment snapshot.  Return no discovered bindings;
            # direct provider names remain governed by the caller's narrow
            # raw-name set.
            return {}
        if not values:
            continue

        # Walk only assignments reachable through an effective value.  This
        # both mirrors override precedence and prevents an unreachable
        # assignment in a lower-precedence file from expanding the narrow
        # capture set.
        if len(pending) > _ROTATION_CAPTURE_MAX_NODES:
            return {}
        queue = list(pending)
        visited: set[str] = set()
        traversed = 0
        while queue:
            traversed += 1
            if traversed > _ROTATION_CAPTURE_MAX_NODES:
                return {}
            name = queue.pop()
            if name in visited:
                continue
            visited.add(name)
            value = values.get(name)
            if value is None:
                continue
            if not override and name in source:
                continue
            source[name] = value
            refs = _token_name_ref_names(value)
            pending.update(refs)
            if len(pending) > _ROTATION_CAPTURE_MAX_NODES:
                return {}
            for ref in refs:
                if ref in environ and ref not in source:
                    source[ref] = environ[ref]
                if ref in values:
                    queue.append(ref)
    return source


def _rotation_command(argv: Sequence[str]) -> tuple[str, str, int] | None:
    """Return provider, subcommand, and first command-argument index."""
    for index in range(len(argv) - 2):
        if argv[index] != "secrets":
            continue
        provider = argv[index + 1].lower()
        if (
            argv[index + 2] in {"setup", "token"}
            and provider in _ROTATION_PROVIDERS
        ):
            return provider, argv[index + 2], index + 3
    return None


def _rotation_provider(argv: Sequence[str]) -> str | None:
    """Return the provider for an exact setup or token command."""
    command = _rotation_command(argv)
    return command[0] if command is not None else None


def _cli_token_env_names(
    argv: Sequence[str],
    *,
    provider: str,
    source: Mapping[str, str],
) -> set[str]:
    """Return explicit OnePassword setup ``--token-env`` names.

    ``argparse`` accepts both separated and equals forms.  Parse only the
    arguments after the exact setup command so a similarly named global or
    unrelated option cannot become part of the pre-dotenv snapshot.  The
    setup handler applies the final argparse value; retaining every explicit
    name here makes repeated options safe while preserving that precedence.
    """
    command = _rotation_command(argv)
    if (
        command is None
        or command[0] not in {"onepassword", "op", "1password"}
        or command[1] != "setup"
        or provider != command[0]
    ):
        return set()

    names: set[str] = set()
    _, _, start = command
    index = start
    while index < len(argv):
        argument = argv[index]
        if argument == "--token-env":
            if index + 1 < len(argv):
                value = argv[index + 1].strip()
                if value and not value.startswith("-"):
                    names.add(value)
                index += 2
                continue
        elif argument.startswith("--token-env="):
            value = argument.split("=", 1)[1].strip()
            if value:
                names.add(value)
        index += 1
    return {name for name in names if name}


def capture_pre_dotenv_rotation_inputs(
    argv: Sequence[str],
    *,
    config: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    dotenv_sources: Sequence[tuple[os.PathLike[str] | str, bool]] = (),
) -> None:
    """Capture only the injected token used by non-TTY setup or rotation.

    This must be called before ``load_hermes_dotenv()``.  The optional raw
    config mapping and the managed-scope provider overlay contribute custom
    provider token names; no other environment values are retained and
    ordinary dotenv precedence is unchanged for every other caller.
    """
    _ROTATION_INPUTS.clear()
    source = os.environ if environ is None else environ
    provider = _rotation_provider(argv)
    if provider is None:
        return

    default_name, config_key = _ROTATION_PROVIDERS[provider]
    raw_names = {default_name}
    raw_names.update(
        _cli_token_env_names(
            argv,
            provider=provider,
            source=source,
        )
    )
    secrets = config.get("secrets") if isinstance(config, Mapping) else None
    provider_name = "onepassword" if provider in {"op", "1password"} else provider
    provider_config = (
        secrets.get(provider_name) if isinstance(secrets, Mapping) else None
    )
    if isinstance(provider_config, Mapping):
        configured_name = provider_config.get(config_key)
        if isinstance(configured_name, str) and configured_name.strip():
            raw_names.add(configured_name.strip())

    # Managed scope wins over user config during normal effective-config
    # loading.  Read only the two provider token-name leaves here so a
    # managed-only override is captured without applying the whole config or
    # copying any broad environment values into the rotation snapshot.
    try:
        from hermes_cli import managed_scope

        managed_config = managed_scope.load_managed_config()
    except Exception:  # noqa: BLE001 — snapshot must never block startup
        managed_config = {}
    managed_secrets = (
        managed_config.get("secrets")
        if isinstance(managed_config, Mapping)
        else None
    )
    managed_provider_config = (
        managed_secrets.get(provider_name)
        if isinstance(managed_secrets, Mapping)
        else None
    )
    if isinstance(managed_provider_config, Mapping):
        managed_name = managed_provider_config.get(config_key)
        if isinstance(managed_name, str) and managed_name.strip():
            raw_names.add(managed_name.strip())

    name_source = _rotation_capture_source(
        raw_names,
        environ=source,
        dotenv_sources=dotenv_sources,
    )
    names = {
        expanded
        for raw_name in raw_names
        for expanded in (_expand_token_name(raw_name, name_source).strip(),)
        if expanded
    }

    # Token-name configuration is persisted input.  Only snapshot values
    # under legal environment names; malformed names must not widen the
    # pre-dotenv capture boundary.
    from agent.secret_sources.base import is_valid_env_name

    for name in names:
        if not name or not is_valid_env_name(name):
            continue
        value = source.get(name)
        if value:
            _ROTATION_INPUTS[name] = value


def get_pre_dotenv_rotation_input(env_var: str) -> str:
    """Return the captured provider value for one configured token name."""
    return _ROTATION_INPUTS.get(env_var, "")


def reset_pre_dotenv_rotation_inputs() -> None:
    """Clear the process-local rotation seam (primarily for tests)."""
    _ROTATION_INPUTS.clear()


def cli_secret_arg_warning(option: str, env_var: str) -> str:
    """Explain why a secret-bearing CLI option is unsafe.

    Keep the warning free of the secret value itself so it is safe to emit in
    command output.  Environment variables are the supported non-interactive
    handoff because they do not put the credential in the command's argv.
    """
    return (
        f"{option} puts the token in process listings, shell history, and CI logs. "
        f"Prefer the masked prompt or provide it through {env_var} in the environment."
    )


def _collect_masked_input(
    read_char: Callable[[], str],
    write: Callable[[str], object],
    prompt: str,
    *,
    mask: str = "*",
) -> str:
    """Read one secret line while writing a mask character per typed char."""
    value: list[str] = []
    write(prompt)

    while True:
        ch = read_char()
        if ch == "":
            write("\r\n")
            raise EOFError
        if ch in _ENTER_CHARS:
            write("\r\n")
            return "".join(value)
        if ch == "\x03":
            write("\r\n")
            raise KeyboardInterrupt
        if ch in _EOF_CHARS:
            write("\r\n")
            raise EOFError
        if ch in _BACKSPACE_CHARS:
            if value:
                value.pop()
                write("\b \b")
            continue
        if ch == "\x1b":
            # Ignore escape itself. Terminals commonly send escape-prefixed
            # navigation/delete sequences; they should not become secret text.
            continue

        value.append(ch)
        if mask:
            write(mask)


def masked_secret_prompt(prompt: str, *, mask: str = "*") -> str:
    """Prompt for a secret while showing masked typing feedback.

    Falls back to ``getpass.getpass`` when stdin/stdout are not interactive or
    when raw terminal handling is unavailable.
    """
    stdin = sys.stdin
    stdout = sys.stdout

    if not _stream_is_tty(stdin) or not _stream_is_tty(stdout):
        return getpass.getpass(prompt)

    if os.name == "nt":
        try:
            return _masked_secret_prompt_windows(prompt, mask=mask)
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            return getpass.getpass(prompt)

    try:
        return _masked_secret_prompt_posix(prompt, mask=mask)
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        return getpass.getpass(prompt)


def _stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _masked_secret_prompt_windows(prompt: str, *, mask: str) -> str:
    import msvcrt

    def read_char() -> str:
        ch = msvcrt.getwch()
        if ch in {"\x00", "\xe0"}:
            msvcrt.getwch()
            return "\x1b"
        return ch

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    return _collect_masked_input(read_char, write, prompt, mask=mask)


def _masked_secret_prompt_posix(prompt: str, *, mask: str) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    def read_char() -> str:
        return sys.stdin.read(1)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        return _collect_masked_input(read_char, write, prompt, mask=mask)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

"""Kitty keyboard protocol — enable enhanced key reporting at startup.

This makes terminals that support the protocol (kitty, WezTerm, Ghostty, foot,
Alacritty 0.13+, iTerm2 3.5+) send unique CSI-u sequences for keys that
otherwise overlap with their unmodified counterparts. The headline win for
Hermes is **Shift+Enter** — without this, terminals send `\\r` for both Enter
and Shift+Enter, so apps can't tell them apart.

We push the enhanced mode at startup and pop it on exit (plus atexit, plus
SIGINT/SIGTERM handlers) so the user's terminal isn't left in enhanced mode
if Hermes crashes. Terminals that don't speak the protocol silently ignore
the escape sequences — no regression, just no Shift+Enter.

Spec: https://sw.kovidgoyal.net/kitty/keyboard-protocol/

Sequences:
    \\x1b[>1u   push enhanced mode (flag 1 = "disambiguate escape codes",
                which is the minimum needed to get CSI-u for Enter+modifiers)
    \\x1b[<u    pop one level off the mode stack

We use the push/pop variant rather than the set/reset variant so that nested
TUIs (vim, less) launched from inside Hermes can push their own modes without
losing ours when they exit.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys
from typing import Optional

# Push: enable disambiguation flag (the minimum we need).
_PUSH = "\x1b[>1u"
# Pop one level. Terminals that didn't honor the push will swallow this too.
_POP = "\x1b[<u"

_active = False
_orig_sigint: Optional[object] = None
_orig_sigterm: Optional[object] = None


def _write(seq: str) -> None:
    """Write directly to /dev/tty if possible, else stdout. Best-effort."""
    try:
        # /dev/tty bypasses any stdout redirection — important during shutdown
        # when stdout may already be closed.
        fd = os.open("/dev/tty", os.O_WRONLY)
        try:
            os.write(fd, seq.encode("ascii"))
        finally:
            os.close(fd)
    except OSError:
        try:
            sys.stdout.write(seq)
            sys.stdout.flush()
        except Exception:
            pass


def enable() -> bool:
    """Push enhanced keyboard mode. Idempotent. Returns True if push was sent.

    Skipped (returns False) when:
      - stdin/stdout aren't TTYs (piped input, CI, tests)
      - already enabled in this process
      - HERMES_DISABLE_KEYBOARD_PROTOCOL env var is set
    """
    global _active, _orig_sigint, _orig_sigterm
    if _active:
        return False
    if os.environ.get("HERMES_DISABLE_KEYBOARD_PROTOCOL"):
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False

    _write(_PUSH)
    _active = True

    # Belt-and-suspenders cleanup. The normal disable() call from cli.py is
    # the primary path; these are for crashes / unexpected exits.
    atexit.register(disable)

    def _sig_handler(signum, frame):  # type: ignore[no-untyped-def]
        disable()
        # Restore + re-raise so default behavior runs (terminate, traceback).
        if signum == signal.SIGINT and callable(_orig_sigint):
            signal.signal(signum, _orig_sigint)  # type: ignore[arg-type]
        elif signum == signal.SIGTERM and callable(_orig_sigterm):
            signal.signal(signum, _orig_sigterm)  # type: ignore[arg-type]
        else:
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        _orig_sigint = signal.getsignal(signal.SIGINT)
        _orig_sigterm = signal.getsignal(signal.SIGTERM)
        # Don't override SIGINT — Hermes' interactive loop relies on it for
        # Ctrl+C interruption. atexit handles the normal exit case; for hard
        # SIGTERM we want to clean up.
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, OSError):
        # signal() can fail in non-main threads or restricted environments.
        # That's fine — atexit still runs.
        pass

    return True


def disable() -> bool:
    """Pop enhanced keyboard mode. Idempotent. Returns True if pop was sent."""
    global _active
    if not _active:
        return False
    _write(_POP)
    _active = False
    return True


def _ensure_keys_member(name: str, value: str):
    """Add a member to prompt_toolkit's `Keys` enum at runtime.

    `KeyBindings.add(key)` validates the key by calling `Keys(key)`, which
    raises `ValueError: Invalid key` for unknown values. To make a new key
    name like `<shift-enter>` bindable we have to extend the enum itself —
    putting the string in `ANSI_SEQUENCES` alone isn't enough, because the
    binding-registration path never consults that dict.

    `Keys` is `class Keys(str, Enum)`, so we mint a `str` instance, attach
    the enum protocol attributes, and splice it into the enum's internal
    maps. Idempotent.
    """
    from prompt_toolkit.keys import ALL_KEYS, Keys

    existing = Keys._value2member_map_.get(value)
    if existing is not None:
        return existing

    member = str.__new__(Keys, value)
    member._name_ = name
    member._value_ = value
    Keys._member_map_[name] = member
    Keys._value2member_map_[value] = member
    if name not in Keys._member_names_:
        Keys._member_names_.append(name)
    # EnumType.__setattr__ blocks adding members, so go through type.__setattr__
    # to make `Keys.ShiftEnter` attribute access work (prompt_toolkit's binding
    # path doesn't need this, but other consumers might).
    try:
        type.__setattr__(Keys, name, member)
    except (TypeError, AttributeError):
        pass
    if value not in ALL_KEYS:
        ALL_KEYS.append(value)
    return member


def register_prompt_toolkit_keys() -> None:
    """Teach prompt_toolkit's input parser about the new CSI-u sequences.

    Two things have to happen for `@kb.add("<shift-enter>")` to work:
      1. `<shift-enter>` must be a real `Keys` enum member (binding-side).
      2. The wire sequence must map to that member (parser-side).

    Idempotent: re-registering is a no-op.
    """
    try:
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    except ImportError:
        return

    shift_enter = _ensure_keys_member("ShiftEnter", "<shift-enter>")

    # Kitty keyboard protocol emits CSI 13;2 u for Shift+Enter. xterm's
    # modifyOtherKeys protocol emits CSI 27;2;13~ — prompt_toolkit ships
    # a default mapping for that to Keys.ControlM (i.e. plain Enter), so
    # we override it to disambiguate when modifyOtherKeys is in use.
    extras: dict[str, object] = {
        "\x1b[13;2u": shift_enter,
        "\x1b[27;2;13~": shift_enter,
    }

    # Kitty's "disambiguate escape codes" flag (>1u, which we push at
    # startup so Shift+Enter works) ALSO routes modified Ctrl+letter and
    # Alt+key combinations through CSI-u instead of their legacy bytes.
    # Without these mappings, Ctrl+C arrives as \x1b[99;5u (unknown to
    # prompt_toolkit) and the kb.add('c-c') binding never fires.
    #
    # Modifier encoding (kitty spec): 1 + (shift=1) + (alt=2) + (ctrl=4).
    # Ctrl alone = 5; Alt alone = 3; Shift+Ctrl = 6; Alt+Ctrl = 7.
    import string as _string
    from prompt_toolkit.keys import Keys as _Keys

    for _ch in _string.ascii_lowercase:
        _member = getattr(_Keys, f"Control{_ch.upper()}", None)
        if _member is not None:
            extras[f"\x1b[{ord(_ch)};5u"] = _member

    # Alt-prefixed keys arrive as a (Escape, key) tuple — that's how
    # prompt_toolkit's existing ANSI_SEQUENCES expresses meta-prefixed
    # sequences (see line "\x1b[1;7u": (Keys.Escape, Keys.Control5)).
    # The emacs key bindings already map (escape, backspace) to
    # backward-kill-word, so this single line restores Option+Delete on
    # macOS (Alt+Backspace) under the disambiguate flag.
    extras["\x1b[127;3u"] = (_Keys.Escape, _Keys.Backspace)

    # Shift+Backspace — kitty disambiguate mode emits CSI 127;2u and
    # prompt_toolkit has no built-in mapping, so the sequence leaks
    # into the input buffer as literal "[127;2u". Map it to plain
    # Backspace so Shift+Backspace just deletes one character (the
    # universal terminal convention; some apps treat it as kill-line
    # but that surprises more users than it pleases).
    extras["\x1b[127;2u"] = _Keys.Backspace

    # Common Alt+letter word-navigation keys (M-b/M-f/M-d) — restore them
    # too so word-jump and kill-word-forward keep working under kitty's
    # disambiguate mode. Emacs bindings register on ('escape', 'b') etc.,
    # i.e. a tuple of (Keys.Escape, literal-char).
    for _ch in _string.ascii_lowercase:
        extras[f"\x1b[{ord(_ch)};3u"] = (_Keys.Escape, _ch)

    for seq, key in extras.items():
        ANSI_SEQUENCES[seq] = key  # type: ignore[assignment]


def is_active() -> bool:
    return _active
